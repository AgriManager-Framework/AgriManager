from agrimanager.checkpoint.publish import find_identity_leaks


def test_release_repo_ids_are_not_leaks():
    text = (
        '{"checkpoint": {"repo": "AgriManager-Framework/schema-t31-dssat-no-think-r2"}, '
        '"suite": {"repo": "AgriManager-Framework/agrimanager-test-suite"}, '
        '"llama": "AgriManager-Framework/Llama-3.2-3B-AgriManager-t32-unified-think"}'
    )
    assert find_identity_leaks(text) == []


def test_project_name_outside_release_repo_ids_is_still_flagged():
    assert find_identity_leaks('{"wandb_project": "agrimanager_t32_unified"}')
    assert find_identity_leaks("/home/someone/AgriManager/runs")


def test_extra_terms_still_apply_to_the_full_text():
    assert find_identity_leaks("AgriManager-Framework/x", extra_terms=["framework/x"])
