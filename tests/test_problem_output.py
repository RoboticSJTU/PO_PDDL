from po_pddl.problem_generation.generator import _write_problem_output


def test_problem_output_creates_parent_directories(tmp_path) -> None:
    output = tmp_path / "nested" / "problem_online.pddl"

    written = _write_problem_output(output, "(define (problem test))\n")

    assert written == output
    assert output.read_text(encoding="utf-8") == "(define (problem test))\n"
