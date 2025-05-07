# import click
# from flask import Flask
# from talkingtomachines.config import DevelopmentConfig
from talkingtomachines.interface.prompt_template import main

# def create_app(config_class=DevelopmentConfig):
#     """Create and configure the Flask application."""
#     app = Flask(__name__)
#     app.config.from_object(config_class)

#     @app.route("/")
#     def home():
#         return "Welcome to the Talking To Machines Platform"

#     return app


# @click.command()
# @click.option("--host", default="127.0.0.1", help="Interface to bind.")
# @click.option("--port", default=5000, show_default=True)
# def main(host: str, port: int):
#     """Run the Talking‑to‑Machines web server."""
#     app = create_app()
#     app.run(host=host, port=port)

if __name__ == "__main__":  # unit‑test & python path safety
    main()
