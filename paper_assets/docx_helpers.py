"""Small lxml helpers for editing the paper's word/document.xml in place."""
import os
import re
import random
import shutil
from copy import deepcopy
from lxml import etree
from PIL import Image

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC_NS = "http://schemas.openxmlformats.org/drawingml/2006/picture"
PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
IMG_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
W = "{%s}" % W_NS
EMU_PER_IN = 914400
TEXT_WIDTH_IN = 4.80  # A4, 2495-twip margins -> 6916 twips


def w(tag):
    return W + tag


def _el(tag, **attrs):
    e = etree.Element(w(tag))
    for k, v in attrs.items():
        e.set(w(k), str(v))
    return e


def _sub(parent, tag, **attrs):
    e = _el(tag, **attrs)
    parent.append(e)
    return e


def para_text(p):
    return "".join(t.text or "" for t in p.iter(w("t")))


# ---------------------------------------------------------------- runs
def run(text, bold=False, italic=False, size=20, sup=False, underline=False):
    r = _el("r")
    rpr = _sub(r, "rPr")
    _sub(rpr, "rFonts", ascii="Times New Roman", hAnsi="Times New Roman", cs="Times New Roman")
    if bold:
        _sub(rpr, "b"); _sub(rpr, "bCs")
    if italic:
        _sub(rpr, "i"); _sub(rpr, "iCs")
    if underline:
        _sub(rpr, "u", val="single")
    if sup:
        _sub(rpr, "vertAlign", val="superscript")
    _sub(rpr, "sz", val=size); _sub(rpr, "szCs", val=size)
    _sub(rpr, "lang", val="en-US")
    t = _sub(r, "t")
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return r


_MARK = re.compile(r"(\*\*.+?\*\*|\*.+?\*)")


def runs_from_markup(text, size=20):
    """'**bold**' and '*italic*' inline markup -> runs."""
    out = []
    for part in _MARK.split(text):
        if not part:
            continue
        if part.startswith("**"):
            out.append(run(part[2:-2], bold=True, size=size))
        elif part.startswith("*"):
            out.append(run(part[1:-1], italic=True, size=size))
        else:
            out.append(run(part, size=size))
    return out


# ---------------------------------------------------------- paragraphs
def para(text="", style="Standard", jc=None, first_line=None, keep_next=False,
         before=None, after=None, size=20, runs=None, left=None):
    p = _el("p")
    ppr = _sub(p, "pPr")
    _sub(ppr, "pStyle", val=style)
    if keep_next:
        _sub(ppr, "keepNext")
    if before is not None or after is not None:
        sp = _sub(ppr, "spacing")
        if before is not None:
            sp.set(w("before"), str(before))
        if after is not None:
            sp.set(w("after"), str(after))
    if first_line is not None or left is not None:
        ind = _sub(ppr, "ind")
        if left is not None:
            ind.set(w("left"), str(left))
        if first_line is not None:
            ind.set(w("firstLine"), str(first_line))
    if jc:
        _sub(ppr, "jc", val=jc)
    for r in (runs if runs is not None else runs_from_markup(text, size=size)):
        p.append(r)
    return p


def h1(text):
    return para(style="0heading1", runs=[run(text, size=24, bold=True)])


def h2(text):
    return para(style="0heading2", runs=[run(text, size=20, italic=True)])


def caption(label, text, after=200):
    return para(style="Standard", jc="center", first_line=0, before=60, after=after,
                runs=[run(label + " ", bold=True, size=18)] + runs_from_markup(text, size=18))


# -------------------------------------------------------------- images
class ImageAdder:
    def __init__(self, unpacked_dir):
        self.dir = unpacked_dir
        self.rels_path = os.path.join(unpacked_dir, "word", "_rels", "document.xml.rels")
        self.rels = etree.parse(self.rels_path)
        os.makedirs(os.path.join(unpacked_dir, "media"), exist_ok=True)
        self.n = 0

    def add(self, src, width_in=TEXT_WIDTH_IN, max_height_in=None):
        self.n += 1
        ext = os.path.splitext(src)[1].lower()
        name = f"case_fig{self.n:02d}{ext}"
        shutil.copy(src, os.path.join(self.dir, "media", name))
        rid = f"rIdCase{self.n:02d}"
        rel = etree.SubElement(self.rels.getroot(), "{%s}Relationship" % PKG_REL)
        rel.set("Id", rid); rel.set("Type", IMG_REL); rel.set("Target", f"/media/{name}")
        with Image.open(src) as im:
            wpx, hpx = im.size
        h_in = width_in * hpx / wpx
        if max_height_in and h_in > max_height_in:
            width_in *= max_height_in / h_in
            h_in = max_height_in
        cx, cy = int(width_in * EMU_PER_IN), int(h_in * EMU_PER_IN)
        doc_id = random.randint(10**8, 2 * 10**9)
        xml = f'''<w:p xmlns:w="{W_NS}" xmlns:wp="{WP_NS}" xmlns:a="{A_NS}" xmlns:pic="{PIC_NS}" xmlns:r="{R_NS}">
<w:pPr><w:pStyle w:val="Standard"/><w:keepNext/><w:spacing w:before="160" w:after="0"/><w:ind w:firstLine="0"/><w:jc w:val="center"/></w:pPr>
<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>
<wp:docPr id="{doc_id}" name="Figure {doc_id}"/><wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>
<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic>
<pic:nvPicPr><pic:cNvPr id="{doc_id}" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr>
<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>
</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'''
        return etree.fromstring(xml.replace("\n", ""))

    def save(self):
        self.rels.write(self.rels_path, xml_declaration=True, encoding="UTF-8", standalone=True)


# -------------------------------------------------------------- tables
def table(header, rows, widths, size=16, bold_first_col=False, align=None):
    """Three-rule (booktabs-style) table. widths in twips; align: list of 'left'/'center'/'right'."""
    tbl = _el("tbl")
    tpr = _sub(tbl, "tblPr")
    _sub(tpr, "tblW", w=sum(widths), type="dxa")
    _sub(tpr, "jc", val="center")
    b = _sub(tpr, "tblBorders")  # schema order: top, left, bottom, right, insideH, insideV
    _sub(b, "top", val="single", sz=8, space=0, color="000000")
    _sub(b, "left", val="nil")
    _sub(b, "bottom", val="single", sz=8, space=0, color="000000")
    for side in ("right", "insideH", "insideV"):
        _sub(b, side, val="nil")
    _sub(tpr, "tblLayout", type="fixed")
    cm = _sub(tpr, "tblCellMar")
    _sub(cm, "left", w=60, type="dxa"); _sub(cm, "right", w=60, type="dxa")
    grid = _sub(tbl, "tblGrid")
    for wd in widths:
        _sub(grid, "gridCol", w=wd)
    align = align or (["left"] + ["center"] * (len(widths) - 1))

    def add_row(cells, is_header=False):
        tr = _sub(tbl, "tr")
        trpr = _sub(tr, "trPr")
        _sub(trpr, "cantSplit")
        if is_header:
            _sub(trpr, "tblHeader")
        for j, (txt, wd) in enumerate(zip(cells, widths)):
            tc = _sub(tr, "tc")
            tcpr = _sub(tc, "tcPr")
            _sub(tcpr, "tcW", w=wd, type="dxa")
            if is_header:
                bb = _sub(tcpr, "tcBorders")
                _sub(bb, "bottom", val="single", sz=4, space=0, color="000000")
            _sub(tcpr, "vAlign", val="center")
            bold = is_header or (bold_first_col and j == 0)
            rs = [run(str(txt), bold=bold, size=size)] if bold else runs_from_markup(str(txt), size=size)
            tc.append(para(style="Standard", jc=align[j], first_line=0, before=20, after=20, runs=rs))

    add_row(header, True)
    for r in rows:
        add_row(r)
    return tbl


def table_caption(label, text):
    return para(style="Standard", jc="center", first_line=0, before=200, after=80, keep_next=True,
                runs=[run(label, size=18)] + [run(" " + text, size=18)])


# ------------------------------------------------------- editing utils
class Doc:
    def __init__(self, unpacked_dir):
        self.dir = unpacked_dir
        self.path = os.path.join(unpacked_dir, "word", "document.xml")
        self.tree = etree.parse(self.path)
        self.body = self.tree.getroot().find(w("body"))
        self.images = ImageAdder(unpacked_dir)

    def paras(self):
        return [e for e in self.body if e.tag == w("p")]

    def find(self, startswith=None, contains=None, nth=0):
        hits = []
        for e in self.body:
            if e.tag != w("p"):
                continue
            t = para_text(e).strip()
            if startswith is not None and t.startswith(startswith):
                hits.append(e)
            elif contains is not None and contains in t:
                hits.append(e)
        if len(hits) <= nth:
            raise KeyError(f"paragraph not found: {startswith or contains!r}")
        return hits[nth]

    def find_by_blip(self, rid):
        for e in self.body:
            for blip in e.iter("{%s}blip" % A_NS):
                if blip.get("{%s}embed" % R_NS) == rid:
                    return e
        raise KeyError(rid)

    @staticmethod
    def insert_after(anchor, *elements):
        cur = anchor
        for el in elements:
            cur.addnext(el)
            cur = el
        return cur

    @staticmethod
    def replace_text(p, old, new):
        """Replace text inside a paragraph; must be contained in a single w:t."""
        for t in p.iter(w("t")):
            if t.text and old in t.text:
                t.text = t.text.replace(old, new)
                t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
                return
        # fall back: collapse paragraph text into the first run
        full = para_text(p)
        if old not in full:
            raise KeyError(f"text not found: {old!r}")
        ts = list(p.iter(w("t")))
        ts[0].text = full.replace(old, new)
        ts[0].set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        for t in ts[1:]:
            t.text = ""

    @staticmethod
    def set_text(p, new):
        ts = list(p.iter(w("t")))
        ts[0].text = new
        ts[0].set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        for t in ts[1:]:
            t.text = ""

    def remove(self, el):
        el.getparent().remove(el)

    def save(self):
        self.images.save()
        self.tree.write(self.path, xml_declaration=True, encoding="UTF-8", standalone=True)
