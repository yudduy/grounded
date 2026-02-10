"""Answer parsing for DeepSeek-R1 and math competition models."""

import re
from typing import Optional


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from DeepSeek-R1 output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


ANSWER_REGEX = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def extract_numeric_answer(text: str) -> str:
    matches = ANSWER_REGEX.findall(text.replace(",", ""))
    if not matches:
        return text.strip()
    result = matches[-1].lstrip("0")
    return result if result else "0"


def last_boxed_only_string(string: str) -> str:
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
    if idx < 0:
        return ""
    brace_idx = string.find("{", idx)
    if brace_idx < 0:
        return ""
    level = 0
    for i in range(brace_idx, len(string)):
        if string[i] == "{":
            level += 1
        elif string[i] == "}":
            level -= 1
            if level == 0:
                return string[idx : i + 1]
    return ""


def clean_answer(s):
    s = s.replace("\\dfrac", "\\frac")
    s = s.replace("x \\in", "")
    s = re.sub(r"\\mathbf\s*{([^}]*)}", r"\1", s)
    s = re.sub(r"\\textbf\s*{([^}]*)}", r"\1", s)
    return s


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]
    left = "\\boxed{"
    if not s.startswith(left):
        return None
    assert s[-1] == "}"
    return clean_answer(s[len(left) : -1])


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a, b = substr[0], substr[1]
                if b != "{":
                    new_str += "{" + a + "}{" + b + "}" + substr[2:]
                else:
                    new_str += "{" + a + "}" + b + substr[2:]
    return new_str


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        a, b = int(a), int(b)
        assert string == "{}/{}".format(a, b)
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except (AssertionError, ValueError):
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    return string


def strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    string = fix_sqrt(string)
    string = string.replace(" ", "")
    string = fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    if string == "5.5":
        string = "\\frac{11}{2}"
    string = fix_a_slash_b(string)
    return string


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False
    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def parse_answer(raw: str) -> str:
    """Parse answer from model output, handling DeepSeek-R1 think blocks."""
    raw = strip_think_blocks(raw)
    boxed = last_boxed_only_string(raw)
    if boxed:
        inner = remove_boxed(boxed)
        if inner is not None:
            return inner.strip()
    m = re.search(r"####\s*(-?[\d,]+\.?\d*)", raw)
    if m:
        return m.group(1).replace(",", "").strip()
    return extract_numeric_answer(raw)


def check_answer(predicted: str, ground_truth: str) -> bool:
    if is_equiv(predicted, ground_truth):
        return True
    try:
        p = int(float(predicted.replace(",", "")))
        g = int(float(ground_truth.replace(",", "")))
        return p == g
    except (ValueError, TypeError):
        return False
