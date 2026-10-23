"""The gateway command line: argument rules and app construction (no server is started)."""

import asyncio

import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("fastapi")

from privgate.gateway.__main__ import build_app, build_parser, main


def parse(*argv):
    return build_parser().parse_args(list(argv))


def test_an_upstream_choice_is_required_and_exclusive():
    with pytest.raises(SystemExit):
        parse("--detector", "regex")
    with pytest.raises(SystemExit):
        parse("--mock-upstream", "--upstream-url", "http://127.0.0.1:1/v1")


def test_non_local_upstream_is_refused_with_exit_code_2(capsys):
    assert main(["--upstream-url", "https://api.example.com/v1"]) == 2
    assert "loopback" in capsys.readouterr().err


def test_fail_open_has_no_command_line_switch():
    with pytest.raises(SystemExit):
        parse("--mock-upstream", "--fail-open")


def test_block_choices_are_entity_types():
    assert parse("--mock-upstream", "--block", "EMAIL", "--block", "IBAN").block == ["EMAIL",
                                                                                      "IBAN"]
    with pytest.raises(SystemExit):
        parse("--mock-upstream", "--block", "NOT_A_TYPE")


def test_mock_upstream_app_masks_and_restores_end_to_end():
    app = build_app(parse("--mock-upstream", "--detector", "regex"))

    async def call():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://g") as client:
            return await client.post("/v1/chat/completions", json={
                "model": "demo", "messages": [{"role": "user", "content": "mail a@b.org"}]})
    reply = asyncio.run(call())
    assert reply.status_code == 200
    assert reply.json()["choices"][0]["message"]["content"] == "ECHO: mail a@b.org"
    assert reply.json()["privgate"]["masked_entities"] == {"EMAIL": 1}
