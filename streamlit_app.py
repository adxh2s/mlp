from __future__ import annotations

from src.ui.app import MLPStreamlitApp

"""Thin entrypoint delegating to the class-based Streamlit app."""


def main() -> None:
    """Run the Streamlit app."""
    MLPStreamlitApp().run()

if __name__ == "__main__":
    main()
