import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from tempfile import TemporaryDirectory
import json

from paling.dataset import parse_markdown_to_sections, chunk_text_by_words, build_datasets
from wonderlib.markdown_xml import markdown_to_xml
from wonderlib.profiling import RarityAnalyzer, profile_document
from wonderlib.git_stats import GitStats, GitCommitEntry

class TestPalingAndWonderLib(unittest.TestCase):
    def test_markdown_parsing(self):
        """
        Verify that markdown headers are parsed correctly into hierarchical sections.
        """
        md_text = (
            "# Title\n"
            "Intro paragraph here.\n"
            "## Section 1\n"
            "Content under section 1.\n"
            "### Sub 1.1\n"
            "Content under sub 1.1.\n"
        )
        sections = parse_markdown_to_sections(md_text, Path("test_note.md"))
        
        self.assertEqual(len(sections), 3)
        self.assertEqual(sections[0]["header"], "Title")
        self.assertEqual(sections[0]["content"], "Intro paragraph here.")
        self.assertEqual(sections[0]["headers_path"], ["Title"])
        
        self.assertEqual(sections[1]["header"], "Section 1")
        self.assertEqual(sections[1]["content"], "Content under section 1.")
        self.assertEqual(sections[1]["headers_path"], ["Title", "Section 1"])
        
        self.assertEqual(sections[2]["header"], "Sub 1.1")
        self.assertEqual(sections[2]["content"], "Content under sub 1.1.")
        self.assertEqual(sections[2]["headers_path"], ["Title", "Section 1", "Sub 1.1"])

    def test_text_chunking(self):
        """
        Verify word-based text chunker sliding window.
        """
        text = "one two three four five six seven eight nine ten"
        chunks = chunk_text_by_words(text, chunk_size=4, overlap=2)
        
        # Expected:
        # Chunk 1: one two three four
        # Chunk 2: three four five six
        # Chunk 3: five six seven eight
        # Chunk 4: seven eight nine ten
        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0], "one two three four")
        self.assertEqual(chunks[1], "three four five six")
        self.assertEqual(chunks[3], "seven eight nine ten")

    def test_markdown_to_xml(self):
        """
        Verify markdown is correctly converted to clean XML.
        """
        md_text = "This is a **bold** paragraph.\n\nAnother paragraph."
        xml_root = markdown_to_xml(md_text)
        
        paragraphs = xml_root.findall("p")
        self.assertEqual(len(paragraphs), 2)
        # Verify formatting tag unwrapping (strong/em should be removed)
        self.assertIn("bold", paragraphs[0].text)

    def test_zipf_scores(self):
        """
        Verify Zipf frequency scoring and clustering.
        """
        analyzer = RarityAnalyzer(token_count=10, model=MagicMock(), tokenizer=MagicMock())
        
        # Test common words Zipf avg (e.g. 'the', 'and', 'of' are very common)
        common_score = analyzer.get_zipf_score("the and of is in to")
        self.assertGreater(common_score, 5.0)
        
        # Verify Zipf clustering
        cluster = analyzer.get_zipf_cluster("the rareconcept and of")
        # 'rareconcept' should fall into bucket 0 (high rarity), others in bucket 2 (low rarity)
        self.assertGreater(cluster[0], 0)
        self.assertGreater(cluster[2], 0)

    def test_profile_document_offline(self):
        """
        Test document profiling using model-free lexical heuristics.
        """
        md_content = "# Test Document\n\nThis is a simple note with some terms."
        profile = profile_document(
            text=md_content,
            title="test-document"
        )
        
        self.assertEqual(profile.title, "test-document")
        self.assertGreater(profile.zipf_avg, 0.0)
        self.assertIsInstance(profile.rare_terms, list)

    def test_profile_document_with_mock_model(self):
        """
        Test document profiling using a mocked LLM analyzer.
        """
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        with patch("wonderlib.profiling.RarityAnalyzer.extract_rare_terms") as mock_extract:
            mock_extract.return_value = ["MockTermA", "MockTermB"]
            
            md_content = "# Test Document\n\nThis is a simple note."
            profile = profile_document(
                text=md_content,
                title="test-document",
                model=mock_model,
                tokenizer=mock_tokenizer
            )
            
            self.assertEqual(profile.title, "test-document")
            self.assertEqual(profile.rare_terms, ["MockTermA", "MockTermB"])

    def test_dataset_compilation_with_rlhf_and_tax(self):
        """
        Test dataset builder combining markdown, RLHF, and Taxonometry data.
        """
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # 1. Create source markdown
            md_dir = tmp_path / "documents"
            md_dir.mkdir()
            with open(md_dir / "note1.md", "w") as f:
                f.write("# Topic A\nContent description.")
                
            # 2. Create RLHF review JSON
            rlhf_dir = tmp_path / "rlhf"
            rlhf_dir.mkdir()
            rlhf_entry = {
                "context": "Context details",
                "questions": [
                    {
                        "question": "What is A?",
                        "answers": [{"answer": "Correct A"}],
                        "approved": True
                    },
                    {
                        "question": "What is B?",
                        "answers": [{"answer": "Wrong B"}],
                        "approved": False # Unapproved, should be ignored
                    }
                ]
            }
            with open(rlhf_dir / "note1-review.json", "w") as f:
                json.dump(rlhf_entry, f)
                
            # 3. Create Taxonometry JSON
            tax_dir = tmp_path / "taxonometry"
            tax_dir.mkdir()
            tax_entry = {
                "title": "note1",
                "zipf_avg": 5.2,
                "rarity_pos": 0.01,
                "rare_terms": ["Term X"]
            }
            with open(tax_dir / "note1-taxonometry.json", "w") as f:
                json.dump(tax_entry, f)
                
            # Compile
            out_dir = tmp_path / "output"
            train_cnt, val_cnt = build_datasets(
                input_dir=str(md_dir),
                output_dir=str(out_dir),
                mode="sections",
                val_split=0.5,
                rlhf_dir=str(rlhf_dir),
                taxonometry_dir=str(tax_dir)
            )
            
            # Expected outputs:
            # - Markdown section: 1 record
            # - RLHF approved QA: 1 record ("What is A?")
            # - Taxonometry profiles: 2 records (overview + rare list)
            # Total = 4 records, split 50/50
            self.assertEqual(train_cnt + val_cnt, 4)
            self.assertEqual(train_cnt, 2)
            self.assertEqual(val_cnt, 2)

if __name__ == "__main__":
    unittest.main()
