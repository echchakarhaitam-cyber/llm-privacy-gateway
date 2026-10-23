"""Run the gateway: ``python -m privgate.gateway --upstream-url http://127.0.0.1:8080/v1``.

Binds to 127.0.0.1 by default. There is deliberately no switch for fail-open mode here: it
exists only as a Python-API option for the RQ9 experiment (docs/GATEWAY.md).
"""

from __future__ import annotations

import argparse
import sys

from privgate.gateway.app import GatewayConfig, create_app
from privgate.gateway.audit import AuditLog
from privgate.gateway.policy import Policy
from privgate.gateway.schema import Limits
from privgate.gateway.upstream import MockUpstream, NonLocalUpstreamError, OpenAICompatibleUpstream
from privgate.spans import EntityType


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m privgate.gateway", description=__doc__)
    where = p.add_mutually_exclusive_group(required=True)
    where.add_argument("--upstream-url",
                       help="base URL of a LOCAL OpenAI-compatible server, "
                            "e.g. http://127.0.0.1:8080/v1")
    where.add_argument("--mock-upstream", action="store_true",
                       help="answer in-process by echoing the masked prompt (demo and tests: "
                            "nothing leaves this process)")
    p.add_argument("--allow-non-local-upstream", action="store_true",
                   help="UNSAFE: allow an upstream that is not on this machine")
    p.add_argument("--detector", default="crf+regex",
                   help="detector configuration, e.g. regex, crf, crf+regex, crf+regex+presidio")
    p.add_argument("--block", action="append", default=[], metavar="LABEL",
                   choices=[e.value for e in EntityType],
                   help="refuse (403) requests containing this entity type; repeatable")
    p.add_argument("--timeout", type=float, default=2.0, help="screening budget per request, s")
    p.add_argument("--max-request-bytes", type=int, default=Limits().max_request_bytes)
    p.add_argument("--max-messages", type=int, default=Limits().max_messages)
    p.add_argument("--audit-log", default=None, help="JSON-lines audit log path (no text in it)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8088)
    return p


def build_app(args: argparse.Namespace):
    """The FastAPI app for parsed arguments; raises NonLocalUpstreamError for a remote upstream."""
    if args.mock_upstream:
        upstream = MockUpstream(record=False)       # a server must not keep every payload
    else:
        upstream = OpenAICompatibleUpstream(
            args.upstream_url, allow_non_local=args.allow_non_local_upstream)
    config = GatewayConfig(
        policy=Policy.blocking(args.block),
        limits=Limits(max_request_bytes=args.max_request_bytes, max_messages=args.max_messages),
        detector_name=args.detector, detector_timeout_s=args.timeout,
        audit_path=args.audit_log)
    return create_app(config, upstream=upstream, audit=AuditLog(args.audit_log))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        app = build_app(args)
    except NonLocalUpstreamError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
