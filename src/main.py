"""Myers' O(ND) diff for Assignment 1.

    python main.py lines A B       line diff of A -> B      (Part A)
    python main.py highlight A B   same, plus changed chars (Part B)

Rough map of the file:
    read_lines      file -> list of lines (bytes)
    diff_marks      two sequences -> which items of A go, which items of B are new
    _myers          the actual shortest-edit-script search used by diff_marks
    _backtrack      turns the saved search rounds back into those marks
    render          marks -> the "-", "+" and " " output lines
    highlight_line  Part B: diff_marks again, but on the characters of one line pair

diff_marks works on any two sequences, so Part A calls it with lists of lines
and Part B calls it with two strings.
"""
import sys
from array import array


def read_lines(path):
    # raw bytes on purpose: no newline translation, no decoding, so "\r\n"
    # files keep their "\r" and weird encodings can't crash us
    with open(path, "rb") as f:
        data = f.read()
    lines = data.split(b"\n")
    if lines[-1] == b"":  # a final newline does not start a new line
        lines.pop()
    return lines


def diff_marks(a, b):
    """Shortest edit script between two sequences (lists of lines or strings).

    Returns (del_a, ins_b), two bytearrays. del_a[i] == 1 means a[i] is
    deleted, ins_b[j] == 1 means b[j] is inserted. What is left unmarked in
    a and in b is the same sequence, so it pairs up in order as keep lines.
    """
    n, m = len(a), len(b)
    del_a = bytearray(n)  # all zeros = nothing deleted yet
    ins_b = bytearray(m)  # all zeros = nothing inserted yet
    if a == b:  # identical inputs, nothing to do
        return del_a, ins_b

    # a common prefix and suffix never need edits, so cut them off first.
    # Real commits usually touch a small part of a big file, so this leaves
    # a tiny middle for the expensive search below.
    lo = 0
    while lo < n and lo < m and a[lo] == b[lo]:
        lo += 1
    hi_a, hi_b = n, m
    while hi_a > lo and hi_b > lo and a[hi_a - 1] == b[hi_b - 1]:
        hi_a -= 1
        hi_b -= 1

    # now only a[lo:hi_a] and b[lo:hi_b] still differ
    if lo == hi_a:  # only insertions are left
        ins_b[lo:hi_b] = b"\x01" * (hi_b - lo)
    elif lo == hi_b:  # only deletions are left
        del_a[lo:hi_a] = b"\x01" * (hi_a - lo)
    else:
        # real search on the middle, then copy its marks back at offset lo.
        # The trimmed prefix and suffix stay 0, meaning "keep".
        d_mid, i_mid = _myers(a[lo:hi_a], b[lo:hi_b])
        del_a[lo:hi_a] = d_mid
        ins_b[lo:hi_b] = i_mid
    return del_a, ins_b


def _myers(a, b):
    """Forward pass of Myers, then backtrack.

    Think of a grid with a along the x axis and b along the y axis. We start
    at (0, 0) and need to get to (n, m). The moves are:
        right     x+1        delete a[x]           costs 1
        down      y+1        insert b[y]           costs 1
        diagonal  x+1, y+1   only if a[x] == b[y]  costs 0 (a "snake")
    The cheapest path is the shortest diff, and its cost d is the edit count.

    Diagonal k is x - y. Every edit changes k by exactly 1, so after d edits
    we can only be on diagonals -d, -d+2, ..., d. v[k] is the furthest x
    reached on diagonal k using d edits. Going further along a diagonal is
    never worse, and y follows from x - k, so v[k] is all we need to store.

    Round d only writes diagonals of parity d and only reads k-1 and k+1,
    which have the other parity, so they still hold round d-1 values. That
    is why one array is enough. After every round we save the diagonals of
    that round (parity d only) so the path can be walked back later.
    """
    n, m = len(a), len(b)
    top = n + m  # worst case: delete all of a, insert all of b
    off = top + 1  # v[k + off] is diagonal k, so k can be negative
    v = [0] * (2 * top + 3)  # the +3 leaves room for the k-1 / k+1 reads at the ends
    trace = []  # trace[d] = diagonals -d, -d+2, ..., d after round d

    for d in range(top + 1):  # d = edits used so far
        for k in range(-d, d + 1, 2):  # every diagonal reachable with d edits
            i = k + off
            # Where did we come from? At k == -d there is no k-1 from the
            # last round, so it has to be a down move from k+1. At k == d it
            # has to be a right move from k-1. In between, pick whichever
            # neighbour got further along its diagonal (ties go to right).
            if k == -d or (k != d and v[i - 1] < v[i + 1]):
                x = v[i + 1]  # move down from k+1: insert b[y]
            else:
                x = v[i - 1] + 1  # move right from k-1: delete a[x]
            y = x - k
            while x < n and y < m and a[x] == b[y]:  # follow the snake
                x += 1
                y += 1
            v[i] = x
            if x >= n and y >= m:  # reached the corner (n, m) with d edits, so d is minimal
                return _backtrack(trace, d, n, m)
        # keep a copy of this round for the backtrack. The slice is exactly
        # diagonals -d, -d+2, ..., d, so it has d + 1 entries.
        trace.append(array("i", v[off - d:off + d + 1:2]))


def _backtrack(trace, d_end, n, m):
    """Walk the saved rounds backwards from (n, m) and mark each edit.

    In round d we ended on diagonal k = x - y. We redo the same decision the
    forward pass made to find out which diagonal of round d-1 we came from,
    and read where that round stopped from trace[d-1]. The step from there
    to here was one edit: a down move (insert) or a right move (delete).
    The snake after it is all keeps, so it needs no marking.
    """
    del_a = bytearray(n)
    ins_b = bytearray(m)
    x, y = n, m
    for d in range(d_end, 0, -1):  # d = 0 is just the opening snake, no edit
        prev = trace[d - 1]  # round d-1: diagonals -(d-1), ..., d-1 step 2
        k = x - y
        # same choice the forward pass made; diagonal kk sits at (kk + d - 1) // 2
        if k == -d or (k != d and prev[(k + d - 2) // 2] < prev[(k + d) // 2]):
            pk = k + 1  # came from k+1 by a down move
            px = prev[(pk + d - 1) // 2]
            py = px - pk
            ins_b[py] = 1  # down move: b[py] was inserted
        else:
            pk = k - 1  # came from k-1 by a right move
            px = prev[(pk + d - 1) // 2]
            py = px - pk
            del_a[px] = 1  # right move: a[px] was deleted
        x, y = px, py  # continue from where round d-1 ended
    return del_a, ins_b


def ranges(marks):
    """'3-5,9-12' style string for the marked positions, '.' if none.

    Each range is start-end with end exclusive, so "3-5" is characters 3 and 4.
    """
    parts = []
    n = len(marks)
    i = 0
    while i < n:
        s = marks.find(1, i)  # start of the next run of marked chars
        if s < 0:
            break
        e = marks.find(0, s)  # first unmarked char after it ends the run
        if e < 0:
            e = n  # the run goes to the end of the line
        parts.append("%d-%d" % (s, e))
        i = e
    return ",".join(parts) if parts else "."


def highlight_line(old, new):
    # utf-8 decode so that indexes count code points; surrogateescape just
    # keeps the function from crashing on bytes that are not valid utf-8
    s = old.decode("utf-8", "surrogateescape")
    t = new.decode("utf-8", "surrogateescape")
    del_s, ins_t = diff_marks(s, t)  # same algorithm, characters instead of lines
    return ("? " + ranges(del_s) + " | " + ranges(ins_t) + "\n").encode("ascii")


def emit_block(out, dels, inss, highlight):
    # one change block: all the removed lines, then all the added lines
    if dels:
        out.append(b"-" + b"\n-".join(dels) + b"\n")
    if not highlight:
        if inss:
            out.append(b"+" + b"\n+".join(inss) + b"\n")
        return
    # Part B: each "+" line is followed by a "?" line that says which
    # characters changed against its partner "-" line. Extra lines on
    # either side with no partner get no "?" line.
    for t, new in enumerate(inss):
        out.append(b"+" + new + b"\n")
        if t < len(dels):  # t-th insertion pairs with t-th deletion
            out.append(highlight_line(dels[t], new))


def render(a, b, del_a, ins_b, highlight):
    out = []
    n, m = len(a), len(b)
    i = j = 0  # i walks through a, j walks through b
    while i < n or j < m:
        # change block: all deletions first, then all insertions
        i0, j0 = i, j
        while i < n and del_a[i]:
            i += 1
        while j < m and ins_b[j]:
            j += 1
        if i > i0 or j > j0:
            emit_block(out, a[i0:i], b[j0:j], highlight)
        # keep lines go on until the next edit on either side. The unmarked
        # lines of a and b are the same sequence, so they advance together.
        next_del = del_a.find(1, i)
        if next_del < 0:
            next_del = n
        next_ins = ins_b.find(1, j)
        if next_ins < 0:
            next_ins = m
        run = min(next_del - i, next_ins - j)
        if run > 0:
            out.append(b" " + b"\n ".join(a[i:i + run]) + b"\n")  # one write for the whole run
            i += run
            j += run
    return b"".join(out)


def main(argv):
    if len(argv) != 4 or argv[1] not in ("lines", "highlight"):
        sys.stderr.write("usage: main.py lines|highlight A B\n")
        return 2
    mode, path_a, path_b = argv[1], argv[2], argv[3]
    try:
        a = read_lines(path_a)
        b = read_lines(path_b)
    except OSError as e:
        # nothing has been printed yet, so stdout stays empty on failure
        sys.stderr.write("cannot read input: %s\n" % e)
        return 2
    del_a, ins_b = diff_marks(a, b)
    data = render(a, b, del_a, ins_b, mode == "highlight")
    sys.stdout.buffer.write(data)  # bytes, so the input is echoed back untouched
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

    