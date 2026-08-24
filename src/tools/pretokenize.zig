const std = @import("std");
const MGT = @import("tokenizer").MGT;
const dataset = @import("mmap_token_dataset");

const CliError = error{
    InvalidArguments,
    InvalidInput,
    TokenOutOfRange,
    InputTooLarge,
};

fn openRead(path: []const u8) !std.fs.File {
    if (std.fs.path.isAbsolute(path)) return std.fs.openFileAbsolute(path, .{ .mode = .read_only });
    return std.fs.cwd().openFile(path, .{ .mode = .read_only });
}

fn openWrite(path: []const u8) !std.fs.File {
    if (std.fs.path.isAbsolute(path)) return std.fs.createFileAbsolute(path, .{ .truncate = true, .read = false });
    return std.fs.cwd().createFile(path, .{ .truncate = true, .read = false });
}

fn readVocab(allocator: std.mem.Allocator, path: []const u8) ![][]const u8 {
    const file = try openRead(path);
    defer file.close();
    const size = std.math.cast(usize, try file.getEndPos()) orelse return CliError.InputTooLarge;
    if (size == 0) return CliError.InvalidInput;
    const bytes = try allocator.alloc(u8, size);
    defer allocator.free(bytes);
    try file.reader().readNoEof(bytes);
    var list = std.ArrayList([]const u8).init(allocator);
    errdefer {
        for (list.items) |item| allocator.free(item);
        list.deinit();
    }
    var lines = std.mem.splitScalar(u8, bytes, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (trimmed.len != 0) try list.append(try allocator.dupe(u8, trimmed));
    }
    return list.toOwnedSlice();
}

fn extractText(allocator: std.mem.Allocator, line: []const u8) ![]const u8 {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, line, .{ .allocate = .alloc_always });
    defer parsed.deinit();
    if (parsed.value != .object) return CliError.InvalidInput;
    const value = parsed.value.object.get("text") orelse return CliError.InvalidInput;
    if (value != .string or value.string.len == 0) return CliError.InvalidInput;
    return try allocator.dupe(u8, value.string);
}

fn writeToken(writer: anytype, value: u32, vocab_size: u32) !void {
    if (value >= vocab_size) return CliError.TokenOutOfRange;
    try writer.writeInt(u32, value, .little);
}

pub fn main() !void {
    var args = std.process.args();
    _ = args.next();
    const input = args.next() orelse return CliError.InvalidArguments;
    const output = args.next() orelse return CliError.InvalidArguments;
    const vocab_path = args.next() orelse return CliError.InvalidArguments;
    const max_seq_raw = args.next() orelse return CliError.InvalidArguments;
    const max_seq = try std.fmt.parseInt(u64, max_seq_raw, 10);
    if (max_seq == 0) return CliError.InvalidArguments;
    const sample_count = if (args.next()) |raw| blk: {
        const value = try std.fmt.parseInt(usize, raw, 10);
        if (value == 0) return CliError.InvalidArguments;
        break :blk value;
    } else null;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    const vocab = try readVocab(allocator, vocab_path);
    defer {
        for (vocab) |item| allocator.free(item);
        allocator.free(vocab);
    }
    var tokenizer = try MGT.init(allocator, vocab, &.{}, null, .english);
    defer tokenizer.deinit();
    const in_file = try openRead(input);
    defer in_file.close();
    var out_file = try openWrite(output);
    defer out_file.close();
    var writer = out_file.writer();
    try writer.writeAll(dataset.magic);
    try writer.writeInt(u32, dataset.version, .little);
    try writer.writeInt(u32, dataset.token_type_u32, .little);
    try writer.writeInt(u64, dataset.header_size, .little);
    const count_position = try out_file.getPos();
    try writer.writeInt(u64, 0, .little);
    const vocabulary_position = try out_file.getPos();
    try writer.writeInt(u64, tokenizer.next_token_id, .little);
    try writer.writeInt(u64, max_seq, .little);
    try writer.writeInt(u64, 0, .little);
    try writer.writeInt(u64, 0, .little);
    var total: u64 = 0;
    var samples_written: usize = 0;
    var buffered = std.io.bufferedReader(in_file.reader());
    while (try buffered.reader().readUntilDelimiterOrEofAlloc(allocator, '\n', 64 * 1024 * 1024)) |line| {
        if (sample_count) |limit| if (samples_written >= limit) {
            allocator.free(line);
            break;
        };
        defer allocator.free(line);
        const text = extractText(allocator, line) catch |err| switch (err) {
            error.OutOfMemory => return err,
            else => try allocator.dupe(u8, line),
        };
        defer allocator.free(text);
        var tokens = std.ArrayList(u32).init(allocator);
        defer tokens.deinit();
        try tokenizer.encode(text, &tokens);
        for (tokens.items) |token| {
            try writeToken(writer, token, tokenizer.next_token_id);
            total = std.math.add(u64, total, 1) catch return CliError.InputTooLarge;
        }
        try writeToken(writer, 3, tokenizer.next_token_id);
        total = std.math.add(u64, total, 1) catch return CliError.InputTooLarge;
        samples_written = std.math.add(usize, samples_written, 1) catch return CliError.InputTooLarge;
    }
    if (sample_count) |limit| if (samples_written != limit) return CliError.InvalidInput;
    try out_file.seekTo(count_position);
    try out_file.writer().writeInt(u64, total, .little);
    try out_file.seekTo(vocabulary_position);
    try out_file.writer().writeInt(u64, tokenizer.next_token_id, .little);
}
