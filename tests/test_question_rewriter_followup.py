from memory.question_rewriter import is_standalone_question, is_follow_up_question, rewrite_question


def test_standalone_questions():
    assert is_standalone_question("Who are the ladies that had relationship with Jose Rizal?")
    assert is_standalone_question("What is the policy that employees must follow?")
    assert is_standalone_question("What did Jose Rizal do with his writings?")
    assert is_standalone_question("Who killed Ferdinand Magellan?")
    assert is_standalone_question("Are there coding rules?")


def test_followup_questions():
    assert is_follow_up_question("What it did?")
    assert is_follow_up_question("Who he was?")
    assert is_follow_up_question("When they arrived?")
    assert is_follow_up_question("What this means?")
    assert is_follow_up_question("What it caused?")
    assert is_follow_up_question("How they escaped?")
    assert is_follow_up_question("Why she left?")
    assert is_follow_up_question("Did Lapu-Lapu kill him?")
    assert is_follow_up_question("What caused it?")


def test_deterministic_rewrite_common_shape():
    history = "User: What is the Treaty of Paris (1898)?\nAssistant: The Treaty of Paris ended the Spanish-American War."

    class DummyLLM:
        def invoke(self, prompt):
            raise AssertionError("LLM should not be needed for this deterministic rewrite")

    assert rewrite_question("What it did?", history, DummyLLM()) == "What did the Treaty of Paris (1898) do?"


def test_compatible_topic_skips_incompatible_latest_topic():
    history = "\n".join([
        "User: What is the Treaty of Paris (1898)?",
        "Assistant: The treaty ceded the Philippines to the United States.",
        "Source: Treaty of Paris (1898) - Wikipedia",
        "User: Who are the ladies that had relationship with Jose Rizal?",
        "Assistant: Gertrude Beckett; Nelly Boustead; Seiko Usui; Leonor Valenzuela; Leonor Rivera.",
        "Source: José Rizal - Wikipedia",
        "User: What it did?",
        "Assistant: The Treaty of Paris (1898) concluded the Spanish-American War.",
        "Source: Treaty of Paris (1898) - Wikipedia",
    ])

    class DummyLLM:
        def invoke(self, prompt):
            raise AssertionError("LLM should not be needed for compatible rewrite")

    assert rewrite_question("when did he died?", history, DummyLLM()) == "when did José Rizal die?"


def test_it_skips_person_group_and_uses_non_person_topic():
    history = "\n".join([
        "User: What is the Treaty of Paris (1898)?",
        "Assistant: It ended the Spanish-American War.",
        "Source: Treaty of Paris (1898) - Wikipedia",
        "User: Who are the ladies that had relationship with Jose Rizal?",
        "Assistant: Gertrude Beckett; Nelly Boustead; Seiko Usui; Leonor Valenzuela; Leonor Rivera.",
        "Source: José Rizal - Wikipedia",
    ])

    class DummyLLM:
        def invoke(self, prompt):
            raise AssertionError("LLM should not be needed for compatible rewrite")

    assert rewrite_question("What it did?", history, DummyLLM()) == "What did Treaty of Paris (1898) do?"
