from typing import Sequence, Tuple


def ascii_radical_sum(
    terms: Sequence[Tuple[int, int, int, int]],
    *,
    max_width: int = 70,
    max_digits: int = 10,
    sqrt_gap: int = 1,
) -> str:
    """
    Return ASCII art for

        sum_j ((p_j + i q_j) / d_j) sqrt(r_j)

    Input:
        terms = [(p1, q1, d1, r1), (p2, q2, d2, r2), ...]

    Features:
        - wraps lines so each output line stays within max_width when possible
        - scales fraction bars and square-root bars
        - skips zero real/imaginary parts
        - pulls a minus sign outside the coefficient if every nonzero
          numerator integer is negative
        - abbreviates long integers as first digits + x10^n
        - omits the fraction when d_j = 1
        - omits the square-root block when r_j = 1
        - writes sqrt(r) instead of 1 sqrt(r)
        - writes i sqrt(r) instead of i 1 sqrt(r)
        - uses at least three dashes for fraction bars
        - returns the string instead of printing it
    """

    if max_width < 10:
        raise ValueError("max_width is too small to make useful ASCII art.")
    if max_digits < 1:
        raise ValueError("max_digits must be positive.")

    def short_int(n: int) -> str:
        sign = "-" if n < 0 else ""
        s = str(abs(n))

        if len(s) <= max_digits:
            return sign + s

        skipped = len(s) - max_digits
        return f"{sign}{s[:max_digits]}x10^{skipped}"

    def centered(s: str, width: int) -> str:
        left = (width - len(s)) // 2
        right = width - len(s) - left
        return " " * left + s + " " * right

    def coefficient_string(
        p: int,
        q: int,
        *,
        parenthesize_complex: bool,
    ) -> tuple[str, str]:
        """
        Return (outer_sign, coefficient_text).

        outer_sign is '+' or '-'. The minus is pulled outside only when all
        nonzero numerator entries are negative.
        """
        nonzero_parts = [x for x in (p, q) if x != 0]

        if not nonzero_parts:
            return "+", ""

        if all(x < 0 for x in nonzero_parts):
            outer_sign = "-"
            p, q = abs(p), abs(q)
        else:
            outer_sign = "+"

        pieces: list[str] = []

        if p != 0:
            pieces.append(short_int(p))

        if q != 0:
            q_abs = abs(q)

            # Write i instead of i 1.
            imag = "i" if q_abs == 1 else "i " + short_int(q_abs)

            if pieces:
                pieces.append((" + " if q > 0 else " - ") + imag)
            else:
                pieces.append(("-" if q < 0 else "") + imag)

        text = "".join(pieces)

        if parenthesize_complex and p != 0 and q != 0:
            text = f"( {text} )"

        return outer_sign, text

    def fraction_block(numerator: str, denominator: str) -> list[str]:
        width = max(len(numerator), len(denominator), 3)

        return [
            centered(numerator, width),
            "-" * width,
            centered(denominator, width),
        ]

    def sqrt_block(radicand: str) -> list[str]:
        width = len(radicand) + 4

        return [
            "   " + "_" * (len(radicand) + 2),
            "  / " + radicand + " ",
            r"\/" + " " * (width - 1),
        ]

    def inline_block(text: str) -> list[str]:
        return [
            " " * len(text),
            text,
            " " * len(text),
        ]

    def join_blocks(left: list[str], right: list[str], gap: int) -> list[str]:
        left_width = max(len(line) for line in left)
        gap_s = " " * gap

        return [
            left[i].ljust(left_width) + gap_s + right[i]
            for i in range(3)
        ]

    def term_block(p: int, q: int, d: int, r: int) -> tuple[str, list[str]]:
        if d == 0:
            raise ZeroDivisionError("Encountered a term with d_j = 0.")

        # Keep denominator positive.
        if d < 0:
            p, q, d = -p, -q, -d

        # Whole term is zero.
        if p == 0 and q == 0:
            return "+", ["", "", ""]

        use_fraction = d != 1
        use_sqrt = r != 1

        sign, coeff = coefficient_string(
            p,
            q,
            parenthesize_complex=not use_fraction,
        )

        # Special case:
        #     1 * sqrt(r) -> sqrt(r)
        # but
        #     1 * sqrt(1) -> 1
        if d == 1 and p == 1 and q == 0 and use_sqrt:
            return sign, sqrt_block(short_int(r))

        coeff_block = (
            fraction_block(coeff, short_int(d))
            if use_fraction
            else inline_block(coeff)
        )

        if not use_sqrt:
            return sign, coeff_block

        return sign, join_blocks(coeff_block, sqrt_block(short_int(r)), sqrt_gap)

    def prefix_lines(sign: str, first: bool) -> list[str]:
        if first:
            if sign == "-":
                return ["  ", "- ", "  "]
            return ["", "", ""]

        return ["   ", f" {sign} ", "   "]

    rendered_terms: list[tuple[str, list[str]]] = []

    for p, q, d, r in terms:
        sign, block = term_block(p, q, d, r)

        if any(line.strip() for line in block):
            rendered_terms.append((sign, block))

    if not rendered_terms:
        return "0"

    groups: list[list[str]] = []
    current = ["", "", ""]
    current_width = 0
    first_term = True

    for sign, block in rendered_terms:
        pref = prefix_lines(sign, first=first_term)
        segment = [pref[i] + block[i] for i in range(3)]
        segment_width = max(len(line) for line in segment)

        if current_width and current_width + segment_width > max_width:
            groups.append(current)

            current = ["", "", ""]
            current_width = 0
            first_term = False

            pref = prefix_lines(sign, first=False)
            segment = [pref[i] + block[i] for i in range(3)]
            segment_width = max(len(line) for line in segment)

        if segment_width > max_width:
            raise ValueError(
                "A single term is wider than max_width. "
                "Increase max_width or reduce max_digits."
            )

        current = [
            current[i] + segment[i]
            for i in range(3)
        ]

        current_width = max(len(line) for line in current)
        first_term = False

    if current_width:
        groups.append(current)

    output_lines: list[str] = []

    for group in groups:
        # If an entire wrapped group is one-line math, do not emit blank
        # top/bottom lines.
        if not group[0].strip() and not group[2].strip():
            output_lines.append(group[1].rstrip())
        else:
            output_lines.extend(line.rstrip() for line in group)

    return "\n".join(output_lines)

def latex_radical_sum(terms):
    if not terms:
        return "0"

    string = ""
    for i, term in enumerate(terms):
        p = term[0]
        q = term[1]
        d = term[2]
        r = term[3]

        if (p <= 0 and q <= 0):
            string += '-'
            p = -p
            q = -q
        elif i != 0:
            string += '+'

        if p != 0 and q!=0:
            numerator = f"{p} + i {q}" if q != 1 else f"{p} + i"
        elif p == 0:
            numerator = f"i {q}" if q != 1 else "i"
        elif q == 0:
            numerator = f"{p}"

        if d!=1:
            string +=(
              r"\frac{"
                +numerator
                +r"}{"   
                +f"{d}"
                +r"}"
            )
        else:
            string += (
                r"\left("
                +numerator
                +r"\right)"
            ) if numerator != "1" else "1"

        if r != 1:
            string += (
                r"\sqrt{"
                +f"{r}"
                +r"}"
            )

    return string
            

        
