"""Local commands: python -m rag_core {serve,status,ask}."""
import argparse
import json

from .service import RAGService


def main():
    parser = argparse.ArgumentParser(description="HR document assistant")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Start the local web application")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    commands.add_parser("status", help="Show the local index status")
    ask = commands.add_parser("ask", help="Ask a question using the local index")
    ask.add_argument("question")
    args = parser.parse_args()

    if args.command == "serve":
        import uvicorn
        uvicorn.run("rag_core.web:create_app", factory=True, host=args.host, port=args.port)
    elif args.command == "status":
        print(json.dumps(RAGService().status(), indent=2, ensure_ascii=False))
    else:
        print(RAGService().ask(args.question)["answer"])


if __name__ == "__main__":
    main()
