# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""Test for the README example code."""


def test_readme_examples() -> None:
    """Test the example code in the README file."""
    with open("README.md", encoding="utf-8") as f:
        lines = f.read().splitlines()

    # Extract code blocks from the README file.
    title = None
    titles = []
    code_blocks = []
    code_block = []
    is_code_block = False
    for line in lines:
        if line.startswith("###"):
            title = line.strip()
        if is_code_block:
            if line == "```":
                if 0 < len(code_block):
                    titles.append(title)
                    code_blocks.append("\n".join(code_block))
                    code_block = []
                is_code_block = False
            else:
                code_block.append(line)
        else:
            if line == "```python":
                is_code_block = True

    assert 0 < len(code_blocks)

    # Execute the code blocks.
    for title, code_block in zip(titles, code_blocks):
        print(f"{title}")
        try:
            exec(code_block)
        except Exception:
            print(code_block)
            assert False
