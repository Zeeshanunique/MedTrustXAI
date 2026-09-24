import sys
import time
from unittest.mock import MagicMock, patch
from pathlib import Path
from PIL import Image

from medtrustxai.config import Config
from medtrustxai.modalities import MODALITIES, normalize_modality
from medtrustxai.app.gradio_app import analyze, chat_followup, PROMPT_PRESETS, _empty_response
from medtrustxai.pipeline.inference import DiagnosticOutput

def p(*args, **kwargs):
    print(*args, **kwargs, flush=True)

def test_fast():
    p("=== STARTING FAST SCENARIO INTEGRATION TESTS ===")
    config_path = "config/default.yaml"
    img = Image.new("RGB", (256, 256), color=(200, 200, 200))
    scenarios_tested = 0

    # 1. Test Empty Image Return Length & Order (All 7 modalities)
    p("\n--- 1. Testing Empty Image across all 7 modalities ---")
    for mod in MODALITIES:
        res = analyze(None, mod, "Full report", "", False, 32, config_path)
        scenarios_tested += 1
        assert len(res) == 6, f"Expected 6 outputs for empty image ({mod}), got {len(res)}"
        chatbot, gradcam, findings, prompt_out, status, faith = res
        assert chatbot == []
        assert gradcam is None
        assert findings == ""
        assert prompt_out == ""
        assert status == "Waiting for image…"
        assert "Enable Grad-CAM" in faith
        p(f"  [OK] {mod}: length 6, status: '{status}'")

    # 2. Test Empty Custom Question (All 7 modalities)
    p("\n--- 2. Testing Empty Custom Prompt across all 7 modalities ---")
    for mod in MODALITIES:
        res = analyze(img, mod, "Custom question", "", False, 32, config_path)
        scenarios_tested += 1
        assert len(res) == 6, f"Expected 6 outputs for custom prompt ({mod}), got {len(res)}"
        chatbot, gradcam, findings, prompt_out, status, faith = res
        assert status == "Enter a custom question."
        p(f"  [OK] {mod}: length 6, status: '{status}'")

    # 3. Test Pipeline Exception Handling Return Structure
    p("\n--- 3. Testing Exception Handling Return Structure ---")
    with patch("medtrustxai.app.gradio_app._get_pipeline") as mock_get_pipe:
        mock_pipe = MagicMock()
        mock_pipe.run.side_effect = RuntimeError("Simulated model failure")
        mock_get_pipe.return_value = mock_pipe
        
        res = analyze(img, "radiology", "Full report", "", False, 32, config_path)
        scenarios_tested += 1
        assert len(res) == 6, f"Expected 6 outputs on exception, got {len(res)}"
        chatbot, gradcam, findings, prompt_out, status, faith = res
        assert "Simulated model failure" in status
        p(f"  [OK] Exception handled gracefully, status: '{status}'")

    # 4. Test Mocked Successful Pipeline Execution across ALL Presets for ALL Modalities
    p("\n--- 4. Testing Output Wiring for ALL Presets across ALL Modalities ---")
    mock_out = DiagnosticOutput(
        modality="radiology",
        findings="No acute cardiopulmonary disease. Heart size normal.",
        prompt="Analyze chest X-ray.",
        gradcam_path=None,
        faithfulness={"faithfulness_score": 0.85, "modality": "radiology", "mentioned_regions": ["lungs"], "predicted_region": "right_lung"},
    )
    
    with patch("medtrustxai.app.gradio_app._get_pipeline") as mock_get_pipe:
        mock_pipe = MagicMock()
        mock_pipe.run.return_value = mock_out
        mock_get_pipe.return_value = mock_pipe

        for mod in MODALITIES:
            presets = PROMPT_PRESETS.get(mod, {})
            for preset_name in presets.keys():
                custom_prompt = "Custom question text?" if preset_name == "Custom question" else ""
                res = analyze(img, mod, preset_name, custom_prompt, False, 32, config_path)
                scenarios_tested += 1
                assert len(res) == 6, f"Mismatch for {mod} / {preset_name}: got {len(res)}"
                chatbot, gradcam, findings, prompt_out, status, faith = res
                assert len(chatbot) == 1
                assert chatbot[0]["role"] == "assistant"
                assert chatbot[0]["content"] == mock_out.findings
                assert findings == mock_out.findings
                assert status.startswith("Done in")
                assert "Score 0.85" in faith
                p(f"  [OK] {mod} -> '{preset_name}': outputs verified")

    # 5. Test Chat Follow-up Edge Cases & State updates
    p("\n--- 5. Testing Chat Followup Scenarios ---")
    # Empty message
    hist, msg, st = chat_followup(img, "radiology", "Findings", "Prompt", [], "", 32, config_path)
    assert st == "Type a question below."
    # Empty findings
    hist, msg, st = chat_followup(img, "radiology", "", "Prompt", [], "Question?", 32, config_path)
    assert st == "Click Analyze first."
    # Empty image
    hist, msg, st = chat_followup(None, "radiology", "Findings", "Prompt", [], "Question?", 32, config_path)
    assert st == "Upload an image first."
    
    # Mocked successful chat
    with patch("medtrustxai.app.gradio_app._get_pipeline") as mock_get_pipe:
        mock_model = MagicMock()
        mock_model.chat.return_value = MagicMock(text="The cardiac silhouette is within normal limits.")
        mock_pipe = MagicMock(model=mock_model)
        mock_get_pipe.return_value = mock_pipe
        
        hist, msg, st = chat_followup(img, "radiology", "Findings text", "Initial prompt", [], "Is the heart enlarged?", 32, config_path)
        scenarios_tested += 4
        assert msg == ""
        assert st.startswith("Replied in")
        assert len(hist) == 2
        assert hist[0] == {"role": "user", "content": "Is the heart enlarged?"}
        assert hist[1] == {"role": "assistant", "content": "The cardiac silhouette is within normal limits."}
        p("  [OK] Chat followup turns and response format verified.")

    p(f"\n==========================================")
    p(f"SUCCESS: All {scenarios_tested} scenarios tested & verified clean!")

if __name__ == "__main__":
    test_fast()
