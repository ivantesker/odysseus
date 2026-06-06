"""CV dataset conversion: COCO/VOC/LabelStudio → YOLO, class-map, split."""
import json

from src.services.cv import convert_data as cd


def test_coco_to_yolo(tmp_path):
    coco = {
        "categories": [{"id": 2, "name": "car"}, {"id": 5, "name": "truck"}],
        "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
        "annotations": [
            {"image_id": 1, "category_id": 2, "bbox": [10, 10, 20, 20]},   # → idx 0
            {"image_id": 1, "category_id": 5, "bbox": [50, 50, 10, 10]},   # → idx 1
        ],
    }
    (tmp_path / "c.json").write_text(json.dumps(coco))
    r = cd.coco_to_yolo(str(tmp_path / "c.json"), str(tmp_path / "out"))
    assert r["images"] == 1 and r["boxes"] == 2
    assert r["class_names"] == ["car", "truck"]
    line = (tmp_path / "out" / "a.txt").read_text().splitlines()[0]
    cls, cx, cy, w, h = line.split()
    assert cls == "0" and abs(float(cx) - 0.2) < 1e-6 and abs(float(w) - 0.2) < 1e-6


def test_voc_to_yolo(tmp_path):
    xml = ('<annotation><size><width>200</width><height>100</height></size>'
           '<object><name>person</name><bndbox><xmin>20</xmin><ymin>10</ymin>'
           '<xmax>60</xmax><ymax>50</ymax></bndbox></object></annotation>')
    vd = tmp_path / "voc"; vd.mkdir()
    (vd / "x.xml").write_text(xml)
    r = cd.voc_to_yolo(str(vd), str(tmp_path / "out"))
    assert r["images"] == 1 and r["class_names"] == ["person"]
    assert (tmp_path / "out" / "x.txt").exists()


def test_labelstudio_to_yolo(tmp_path):
    ls = [{"data": {"image": "/up/img1.jpg"}, "annotations": [{"result": [
        {"type": "rectanglelabels", "value": {"x": 10, "y": 20, "width": 30, "height": 40,
                                              "rectanglelabels": ["bike"]}}]}]}]
    (tmp_path / "ls.json").write_text(json.dumps(ls))
    r = cd.labelstudio_to_yolo(str(tmp_path / "ls.json"), str(tmp_path / "out"))
    assert r["images"] == 1 and r["class_names"] == ["bike"]
    cls, cx, cy, w, h = (tmp_path / "out" / "img1.txt").read_text().split()
    assert cls == "0" and abs(float(cx) - 0.25) < 1e-6 and abs(float(w) - 0.30) < 1e-6


def test_auto_class_map_case_insensitive():
    cm = cd.auto_class_map([
        {"name": "A", "class_names": ["car", "Truck"]},
        {"name": "B", "class_names": ["truck", "bus"]},
    ])
    assert cm["unified_names"] == ["car", "Truck", "bus"]
    assert cm["remaps"][0]["map"] == {0: 0, 1: 1}
    assert cm["remaps"][1]["map"] == {0: 1, 1: 2}   # B's truck → unified Truck idx


def test_stratified_split_no_source_leakage(tmp_path):
    lbl = tmp_path / "labels"; lbl.mkdir()
    for i in range(20):
        src = "vidA" if i < 10 else "vidB"
        (lbl / f"{src}_frame{i}.txt").write_text("0 .5 .5 .2 .2\n")
    r = cd.stratified_split(str(lbl), val_frac=0.5, source_regex=r"(vid[AB])", class_names=["car"])
    assert r["n_train"] + r["n_val"] == 20
    assert r["n_sources"] == 2 and r["leak_free"] is True
    # whole source goes to one split → train/val are each a single video
    assert r["n_train"] == 10 and r["n_val"] == 10


def test_convert_annotations_unknown_format(tmp_path):
    assert cd.convert_annotations("x", "bogus", str(tmp_path))["exit_code"] == 1
