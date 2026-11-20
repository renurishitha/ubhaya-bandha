"""Tests for real GEO 10x .mtx ingestion and stratified per-cell-type subsampling."""

import tempfile
import unittest
from pathlib import Path

from bspai.data import load_candidate_genes, load_geo_dataset
from bspai.preprocess import preprocess_scrna_dataset


class TestGeoLoader(unittest.TestCase):
    def test_load_10x_mtx_bundle_with_subsampling(self) -> None:
        genes = load_candidate_genes()[:10]
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp)
            # 3 features (CD274, CTLA4, PDCD1), 6 cells across 2 cell types (3 per type)
            (raw_dir / "features.tsv").write_text(
                "\n".join(
                    [
                        f"ENSG001\t{genes[0].gene}",
                        f"ENSG002\t{genes[1].gene}",
                        f"ENSG003\t{genes[2].gene}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            barcodes = [f"CELL_{i}" for i in range(1, 7)]
            (raw_dir / "barcodes.tsv").write_text("\n".join(barcodes) + "\n", encoding="utf-8")
            (raw_dir / "metadata.csv").write_text(
                "barcode,tissue,cell_type,pct_counts_mt\n"
                "CELL_1,tumor_lung,exhausted_cd8_t,4.0\n"
                "CELL_2,tumor_lung,exhausted_cd8_t,5.0\n"
                "CELL_3,normal_lung,exhausted_cd8_t,6.0\n"
                "CELL_4,tumor_lung,epithelial,3.0\n"
                "CELL_5,tumor_lung,epithelial,4.0\n"
                "CELL_6,normal_lung,epithelial,5.0\n",
                encoding="utf-8",
            )
            # Matrix Market: 3 features x 6 cells, 12 entries
            (raw_dir / "matrix.mtx").write_text(
                "%%MatrixMarket matrix coordinate real general\n"
                "3 6 12\n"
                "1 1 12.0\n2 1 8.0\n"
                "1 2 10.0\n3 2 6.0\n"
                "1 3 2.0\n2 3 1.0\n"
                "1 4 15.0\n3 4 4.0\n"
                "2 5 9.0\n3 5 7.0\n"
                "1 6 3.0\n2 6 2.0\n",
                encoding="utf-8",
            )

            ds = load_geo_dataset(
                raw_dir=raw_dir,
                candidate_genes=genes,
                dataset_id="GSE131907_TEST",
                max_cells_per_type=2,
                seed=42,
            )
            self.assertEqual(ds.data_mode, "REAL_GEO")
            self.assertEqual(ds.dataset_id, "GSE131907_TEST")
            # 2 cell types * max 2 per type = 4 cells retained
            self.assertEqual(len(ds.cells), 4)
            prep = preprocess_scrna_dataset(ds, min_genes=1, max_mito_pct=20.0, use_scanpy=False)
            self.assertEqual(prep.qc_stats["retained_cells"], 4)


if __name__ == "__main__":
    unittest.main()
