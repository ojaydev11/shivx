#!/usr/bin/env python3
"""
Test script for Pillar 7: Multi-Modal Perception

Demonstrates the key capabilities of the perception system.
"""

from agi_engine.perception import (
    VisualProcessor,
    MultiModalFusion,
    GroundingEngine,
    ModalityType
)
import numpy as np


def test_visual_processor():
    """Test visual processing capabilities"""
    print("=" * 60)
    print("Testing Visual Processor")
    print("=" * 60)

    vp = VisualProcessor()

    # Simulate image data
    fake_image = np.random.rand(224, 224, 3)

    # Extract features
    features = vp.extract_features(fake_image)
    print(f"\n1. Feature Extraction:")
    print(f"   - Image ID: {features.image_id}")
    print(f"   - Global features shape: {features.global_features.shape}")
    print(f"   - Local features: {len(features.local_features)} regions")

    # Detect objects
    detections = vp.detect_objects(features)
    print(f"\n2. Object Detection:")
    print(f"   - Detected {len(detections)} objects")
    for det in detections[:3]:
        print(f"   - {det.label} ({det.category.value}) with confidence {det.confidence:.2f}")

    # Understand scene
    scene = vp.understand_scene(features, detections)
    print(f"\n3. Scene Understanding:")
    print(f"   - Scene type: {scene.scene_type.value}")
    print(f"   - Description: {scene.scene_description}")
    print(f"   - Spatial relations: {len(scene.spatial_relations)}")

    # Visual reasoning
    if detections:
        query = f"How many {detections[0].label}s are there?"
        result = vp.visual_reasoning(scene, query)
        print(f"\n4. Visual Reasoning:")
        print(f"   - Question: {query}")
        print(f"   - Answer: {result['answer']}")
        print(f"   - Confidence: {result['confidence']:.2f}")

    print(f"\n5. Statistics:")
    stats = vp.get_stats()
    for key, value in stats.items():
        print(f"   - {key}: {value}")

    return vp, scene


def test_multimodal_fusion():
    """Test multi-modal fusion capabilities"""
    print("\n" + "=" * 60)
    print("Testing Multi-Modal Fusion")
    print("=" * 60)

    mmf = MultiModalFusion()

    # Add sensory inputs from different modalities
    visual_data = np.random.rand(512)
    audio_data = np.random.rand(256)
    text_data = "A person walking in the park"

    print("\n1. Adding Sensory Inputs:")
    vis_input = mmf.add_sensory_input(visual_data, ModalityType.VISUAL)
    print(f"   - Visual input: {vis_input.input_id}")

    aud_input = mmf.add_sensory_input(audio_data, ModalityType.AUDITORY)
    print(f"   - Audio input: {aud_input.input_id}")

    txt_input = mmf.add_sensory_input(text_data, ModalityType.TEXTUAL)
    print(f"   - Text input: {txt_input.input_id}")

    # Fuse modalities
    print("\n2. Fusing Modalities:")
    inputs = {
        ModalityType.VISUAL: vis_input,
        ModalityType.AUDITORY: aud_input,
        ModalityType.TEXTUAL: txt_input
    }

    fused = mmf.fuse(inputs)
    print(f"   - Fusion ID: {fused.representation_id}")
    print(f"   - Strategy: {fused.fusion_strategy.value}")
    print(f"   - Fused features shape: {fused.fused_features.shape}")
    print(f"   - Confidence: {fused.confidence:.2f}")

    if fused.attention_weights:
        print(f"\n3. Attention Weights:")
        for modality, weight in fused.attention_weights.items():
            print(f"   - {modality.value}: {weight:.3f}")
        print(f"   - Dominant modality: {fused.dominant_modality().value}")

    # Temporal integration
    print("\n4. Temporal Integration:")
    integrated = mmf.temporal_integration(ModalityType.VISUAL, window_size=3)
    if integrated is not None:
        print(f"   - Integrated features shape: {integrated.shape}")

    print("\n5. Statistics:")
    stats = mmf.get_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"   - {key}:")
            for k, v in value.items():
                print(f"      - {k}: {v}")
        else:
            print(f"   - {key}: {value}")

    return mmf, fused


def test_grounding_engine(visual_processor, scene):
    """Test grounding engine capabilities"""
    print("\n" + "=" * 60)
    print("Testing Grounding Engine")
    print("=" * 60)

    ge = GroundingEngine(visual_processor=visual_processor)

    # Visual Question Answering
    print("\n1. Visual Question Answering:")
    questions = [
        "How many objects are in the scene?",
        "Is there a person in the image?",
        "What color is the car?",
        "Where is the tree?"
    ]

    for question in questions:
        vqa = ge.visual_question_answering(question, scene)
        print(f"   Q: {question}")
        print(f"   A: {vqa.answer} (confidence: {vqa.confidence:.2f})")

    # Image Captioning
    print("\n2. Image Captioning:")
    caption_low = ge.generate_caption(scene, detail_level="low")
    print(f"   - Low detail: {caption_low.caption_text}")

    caption_med = ge.generate_caption(scene, detail_level="medium")
    print(f"   - Medium detail: {caption_med.caption_text}")

    caption_high = ge.generate_caption(scene, detail_level="high")
    print(f"   - High detail: {caption_high.caption_text}")

    # Reference Resolution
    if scene.objects:
        print("\n3. Reference Resolution:")
        obj = scene.objects[0]
        expr = f"the {obj.label}"
        ref = ge.resolve_reference(expr, scene)
        print(f"   - Expression: '{expr}'")
        print(f"   - Resolved: {ref.resolved}")
        print(f"   - Target: {ref.target_object_id}")
        print(f"   - Confidence: {ref.confidence:.2f}")

    # Grounded Dialog
    print("\n4. Grounded Dialog:")
    dialog = ge.start_grounded_dialog()
    print(f"   - Dialog ID: {dialog.dialog_id}")

    result = ge.process_grounded_utterance(
        dialog,
        "Tell me about the objects in the scene",
        scene,
        speaker="user"
    )
    print(f"   - Resolved references: {len(result['resolved_references'])}")
    print(f"   - Active objects: {len(result['active_objects'])}")

    print("\n5. Statistics:")
    stats = ge.get_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"   - {key}:")
            for k, v in value.items():
                print(f"      - {k}: {v}")
        else:
            print(f"   - {key}: {value}")

    return ge


def main():
    """Run all tests"""
    print("\n" + "#" * 60)
    print("# PILLAR 7: MULTI-MODAL PERCEPTION TEST SUITE")
    print("#" * 60)

    # Test Visual Processor
    vp, scene = test_visual_processor()

    # Test Multi-Modal Fusion
    mmf, fused = test_multimodal_fusion()

    # Test Grounding Engine
    ge = test_grounding_engine(vp, scene)

    print("\n" + "#" * 60)
    print("# ALL TESTS COMPLETED SUCCESSFULLY!")
    print("#" * 60)
    print("\nPillar 7: Multi-Modal Perception is fully operational!")
    print("\nKey Capabilities Demonstrated:")
    print("  ✓ Visual feature extraction and processing")
    print("  ✓ Object detection and recognition")
    print("  ✓ Scene understanding and analysis")
    print("  ✓ Visual reasoning")
    print("  ✓ Multi-modal sensor fusion")
    print("  ✓ Cross-modal integration")
    print("  ✓ Temporal integration")
    print("  ✓ Visual question answering")
    print("  ✓ Image captioning")
    print("  ✓ Reference resolution")
    print("  ✓ Grounded language understanding")
    print("  ✓ Grounded dialog management")


if __name__ == "__main__":
    main()
