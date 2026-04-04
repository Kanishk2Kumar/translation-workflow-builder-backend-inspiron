from nodes.document_upload import DocumentUploadNode
from nodes.document_intelligence_ocr import DocumentIntelligenceOCRNode
from nodes.document_parser import DocumentParserNode       # new
from nodes.document_rebuilder import DocumentRebuilderNode  # new
from nodes.google_vision_ocr import GoogleVisionOCRNode
from nodes.ocr_confidence_gate import OCRConfidenceGateNode
from nodes.phi_detector import PHIDetectorNode             # new
from nodes.phi_restore import PHIRestoreNode               # new
from nodes.glossary import GlossaryNode
from nodes.google_translate import AzureTranslateNode, GoogleTranslateNode
from nodes.rag_tm import RAGNode
from nodes.llm_agent import LLMAgentNode
from nodes.compliance_enforcer import ComplianceEnforcerNode
from nodes.compliance import ComplianceNode
from nodes.output import OutputNode

NODE_REGISTRY: dict[str, type] = {
    "document_upload":    DocumentUploadNode,
    "document_intelligence_ocr": DocumentIntelligenceOCRNode,
    "document_parser":    DocumentParserNode,    # new
    "google_vision_ocr":  GoogleVisionOCRNode,
    "ocr_confidence_gate": OCRConfidenceGateNode,
    "phi_detector":       PHIDetectorNode,     # new
    "glossary":           GlossaryNode, 
    "document_rebuilder": DocumentRebuilderNode, # new
    "rag_tm":             RAGNode,
    "vector_db":          RAGNode,
    "compliance_enforcer": ComplianceEnforcerNode,
    "llm_agent":          LLMAgentNode,
    "translation":        LLMAgentNode,
    "google_translate":   GoogleTranslateNode,
    "azure_translate":    AzureTranslateNode,
    "phi_restore":        PHIRestoreNode, 
    "compliance":         ComplianceNode,
    "output":             OutputNode,
    "document_output":    OutputNode,
}
