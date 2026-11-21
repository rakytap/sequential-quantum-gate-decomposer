import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

path=Path(__file__).resolve().parents[2]/"examples/decomposition/build_symbolic_atlas.py"
spec=importlib.util.spec_from_file_location("build_symbolic_atlas",path)
builder=importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_split_pages_resolve_the_index_before_independent_compilation():
    source=r"""\documentclass{article}\begin{document}
\hyperref[gate-SWAP]{SWAP} p.~\pageref{gate-SWAP}
\newpage\phantomsection\label{gate-SWAP}
\pdfbookmark[0]{SWAP}{bookmark-SWAP}
\end{document}"""
    pages,destinations=builder.split_document(source)
    assert len(pages)==2 and destinations=={"gate-SWAP":1}
    assert pages[0]["links"]==["gate-SWAP"]
    assert r"\hyperlink{gate-SWAP}{SWAP} p.~2" in pages[0]["source"]
    assert r"\setcounter{page}{2}" in pages[1]["source"]
    assert pages[1]["bookmarks"]==[("0","SWAP")]
    with pytest.raises(ValueError,match="missing"):
        builder.split_document(source.replace(r"\label{gate-SWAP}",""))


def test_assembled_index_links_target_the_actual_merged_page(tmp_path):
    pypdf=pytest.importorskip("pypdf")
    from pypdf.annotations import Link
    paths=[]
    for index in range(2):
        writer=pypdf.PdfWriter()
        writer.add_blank_page(width=200,height=200)
        if index==0:writer.add_annotation(0,Link(rect=(10,10,80,30),target_page_index=0))
        path=tmp_path/f"page-{index}.pdf"
        with path.open("wb") as stream:writer.write(stream)
        paths.append(path)
    pages=[dict(links=["gate-SWAP"],bookmarks=[]),dict(links=[],bookmarks=[("0","SWAP")])]
    report=builder.assemble(paths,pages,{"gate-SWAP":1},tmp_path/"atlas.pdf")
    reader=pypdf.PdfReader(tmp_path/"atlas.pdf")
    annotation=reader.pages[0]["/Annots"][0].get_object()
    assert annotation["/Dest"][0]==reader.pages[1].indirect_reference
    assert reader.outline[0].title=="SWAP"
    assert reader.get_destination_page_number(reader.outline[0])==1
    assert report["pages"]==2 and report["repaired_index_links"]==1


def test_dense_page_fallback_keeps_exact_fragments_fonts_and_caches_vectors(tmp_path, monkeypatch):
    circuits=tmp_path/"circuits"
    circuits.mkdir()
    fragment=r"\Qcircuit { & \gate{R_z(\theta)} & \qw }"
    (circuits/"rotation.tex").write_text(fragment)
    document=r"""\documentclass[10pt]{article}
\usepackage[a3paper]{geometry}
\AtBeginDocument{\fontsize{14}{17}\selectfont}
\begin{document}\input{circuits/rotation.tex}\input{circuits/rotation.tex}\end{document}"""
    calls=[]
    def run(command, **kwargs):
        source=Path(command[-1])
        calls.append(source.read_text())
        source.with_suffix(".pdf").touch()
        return SimpleNamespace(returncode=0,stdout="",stderr="")
    monkeypatch.setattr(builder.subprocess,"run",run)
    rendered=builder.vectorize_circuits(document,"tectonic",tmp_path,tmp_path/"pages",30)
    assert len(calls)==1
    assert fragment in calls[0]
    assert r"\AtBeginDocument{\fontsize{14}{17}\selectfont}" in calls[0]
    assert r"\documentclass[border=1pt]{standalone}" in calls[0]
    assert "geometry" not in calls[0]
    assert rendered.count(r"\includegraphics{")==2 and r"\input{" not in rendered
    assert builder.vectorize_circuits(document,"tectonic",tmp_path,tmp_path/"pages",30)==rendered
    assert len(calls)==1


def test_blank_pdf_backend_failure_is_retried_before_stamping(tmp_path,monkeypatch):
    calls=[]
    def run(command,**kwargs):
        calls.append(command)
        source=Path(command[-1])
        if len(calls)==1:
            return SimpleNamespace(returncode=-11,stdout="note: Running xdvipdfmx ...",stderr="")
        source.with_suffix(".pdf").touch()
        return SimpleNamespace(returncode=0,stdout="completed",stderr="")
    monkeypatch.setattr(builder.subprocess,"run",run)
    page=dict(source=r"\documentclass{article}\begin{document}test\end{document}")
    index,output=builder.compile_page(("tectonic",tmp_path,"gates",0,page,30))
    assert index==0 and output.exists() and output.with_suffix(".sha256").exists()
    assert len(calls)==2
