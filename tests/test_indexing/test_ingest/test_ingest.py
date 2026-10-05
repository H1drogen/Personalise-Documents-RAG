from indexing.ingest import load_context_docs


def test_load_context_docs_recursively_loads_all_text_files(tmp_path):
    (tmp_path / "root.txt").write_text("root", encoding="utf-8")
    (tmp_path / "root2.txt").write_text("root2", encoding="utf-8")

    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "child.txt").write_text("child", encoding="utf-8")

    docs = load_context_docs(str(tmp_path))

    assert [doc.page_content for doc in docs] == ["child", "root", "root2"]


def test_load_context_docs_accepts_a_single_file(tmp_path):
    file_path = tmp_path / "single.txt"
    file_path.write_text("single", encoding="utf-8")

    docs = load_context_docs(str(file_path))

    assert len(docs) == 1
    assert docs[0].page_content == "single"