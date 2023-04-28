#!/usr/bin/env python3

import json
import logging
from time import time

from confparser import createParser
from flask import Flask, Response, abort, json, request
from serving import GunicornServing, GeventServing
from swagger import setupSwaggerUI

from diarization.processing import diarizationworker, USE_GPU

app = Flask("__diarization-serving__")

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
)
logger = logging.getLogger("__diarization-serving__")


@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return json.dumps({"healthcheck": "OK"}), 200


@app.route("/oas_docs", methods=["GET"])
def oas_docs():
    return "Not Implemented", 501



@app.route("/diarization", methods=["POST"])
def transcribe():
    try:
        logger.info("Diarization request received")

        # get response content type
        logger.debug(request.headers.get("accept").lower())
        if not request.headers.get("accept").lower() == "application/json":
            raise ValueError("Not accepted header")

        # get input file
        if "file" in request.files.keys():
            spk_number = request.form.get("spk_number", None)
            if spk_number is not None:
                spk_number = int(spk_number)
            max_spk_number = request.form.get("max_speaker", None)
            if max_spk_number is not None:
                max_spk_number = int(max_spk_number)
            start_t = time()
        else:
            raise ValueError("No audio file was uploaded")
    except ValueError as error:
        return str(error), 400
    except Exception as e:
        logger.error(e)
        return "Server Error: {}".format(str(e)), 500

    # Diarization
    try:
        result = diarizationworker.run(
            request.files["file"], number_speaker=spk_number, max_speaker=max_spk_number
        )
    except Exception as e:
        return "Diarization has failed: {}".format(str(e)), 500

    response = diarizationworker.format_response(result)
    logger.debug("Diarization complete (t={}s)".format(time() - start_t))

    return response, 200


# Rejected request handlers
@app.errorhandler(405)
def method_not_allowed(error):
    return "The method is not allowed for the requested URL", 405


@app.errorhandler(404)
def page_not_found(error):
    return "The requested URL was not found", 404


@app.errorhandler(500)
def server_error(error):
    logger.error(error)
    return "Server Error", 500


if __name__ == "__main__":
    logger.info("Startup...")

    parser = createParser()
    args = parser.parse_args()
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    try:
        # Setup SwaggerUI
        if args.swagger_path is not None:
            setupSwaggerUI(app, args)
            logger.debug("Swagger UI set.")
    except Exception as e:
        logger.warning("Could not setup swagger: {}".format(str(e)))

    if USE_GPU: # TODO: get rid of this?
        serving_type = GeventServing
        logger.debug("Serving with gevent")
    else:
        serving_type = GunicornServing
        logger.debug("Serving with gunicorn")

    serving = serving_type(
        app,
        {
            "bind": "{}:{}".format("0.0.0.0", args.service_port),
            "workers": args.workers,
            "timeout": 3600,
        },
    )
    logger.info(args)
    try:
        serving.run()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(str(e))
        logger.critical("Service is shut down (Error)")
        exit(e)
