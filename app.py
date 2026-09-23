import os

from flask import Flask, render_template, request

from eo import (
    get_satellite_image,
    find_historical_images
)

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    historical = None
    error = None

    if request.method == "POST":

        try:

            latitude = float(
                request.form["latitude"]
            )

            longitude = float(
                request.form["longitude"]
            )

            # Current image
            result = get_satellite_image(
                latitude,
                longitude
            )

            # Historical images
            historical = find_historical_images(
                latitude,
                longitude
            )

            print("HISTORICAL RESULT:")
            print(historical)

        except Exception as e:

            error = str(e)

            print("ERROR:")
            print(error)

    return render_template(
        "index.html",
        result=result,
        historical=historical,
        error=error
    )


if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )
