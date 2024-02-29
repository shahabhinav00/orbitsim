import requests
import numpy as np
import datetime

# loads data from Horizons
# no need to run this


def datetime_to_string(time):
    return (
        f"{time.year}-{time.month}-{time.day}-{time.hour}:{time.minute}:{time.second}"
    )


BASE_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
# TODO: set time to the time we present
START_TIME = "2023-01-01-12:00:00"
STOP_TIME = "2023-01-01-12:00:01"


#OBSERVED_TIME = datetime_to_string(datetime.datetime.utcnow())


def fill_request_url(**kwargs):
    url = BASE_URL
    if len(kwargs) != 0:
        url += "?"
        arg_list = []
        for key, value in kwargs.items():
            arg_list.append(key + "=" + value)
        url += "&".join(arg_list)
    return url


def get_response_raw_text(id_code, t0=START_TIME, t1=STOP_TIME):
    url = fill_request_url(
        format="text",
        EPHEM_TYPE="VECTORS",
        OBJ_DATA="NO",
        COMMAND=f"'{id_code}'",
        VEC_TABLE="2",
        CAL_TYPE="GREGORIAN",
        CSV_FORMAT="YES",
        VEC_LABELS="NO",
        START_TIME=str(t0),
        STOP_TIME = str(t1),
        STEP_SIZE="1",
        CENTER="500@0",  # uncomment line for sol barycenter frame of reference, otherwise earth
    )
    return requests.get(url).text.replace("\t", " ")


# still doesn't work...
def get_mass(text):
    possible_keys = [
        "Mass x10^",
        "Mass, x10^",
        "Mass 10^",
        "Mass, 10^",
    ]

    for key in possible_keys:
        if key in text:
            break

    lines = text.split("\n")
    for line in lines:
        if key in line:
            break

    print(key)
    print(line)

    line = line[line.index(key) + len(key) :]

    exponent = int(line[: line.index(" ")])

    if "." in line:
        line_value = line[line.index(".") - 1 :]

    elif "~" in line:
        line_value = line[line.index("~") + 1]

    if "+-" in line_value:
        value = float(line_value[: line_value.index("+-")])

    elif " " in line_value:
        value = float(line_value[: line_value.index(" ")])

    else:
        value = float(line_value)

    return value * 10**exponent


def get_state_vector(text):
    line = text[text.index("$$SOE") + 6 :]
    line = line[: line.index("\n")]

    result = np.empty(6)

    nums = line.split(",")[2:]
    for i in range(6):
        result[i] = float(nums[i].replace(" ", ""))

    return result


def request_state_vector(id_code, time):
    text = get_response_raw_text(id_code, datetime_to_string(time))


if __name__ == "__main__":
    selected_planets = [
        399,  # earth
        301,  # moon
        10,  # sun
        199,  # other planets, in order
        299,
        499,
        599,
        699,
        799,
        899,
        # 134340, # pluto
    ]

    num = len(selected_planets)
    SOL_MASSES = np.array(
        [
            5.97219 * (10**24),  # earth
            7.349 * (10**22),  # moon
            1988500.0 * (10**24),  # sun
            3.302 * (10**23),  # mercury
            48.685 * (10**23),  # venus
            6.4171 * (10**23),  # mars
            189818.722 * (10**22),  # jupiter
            5.6834 * (10**26),  # saturn
            86.813 * (10**24),  # uranus
            102.409 * (10**24),  # neptune
            # 1.307 * (10 ** 22) # pluto
        ]
    )
    SOL_STATE_VECTORS = np.empty((num, 2, 3))

    for i, id_num in enumerate(selected_planets):
        text = get_response_raw_text(str(id_num))
        print(text)
        SOL_STATE_VECTORS[i] = np.reshape(get_state_vector(text), (2, 3))

    np.save("data/sol_masses.npy", SOL_MASSES)
    np.save("data/sol_state_vectors.npy", SOL_STATE_VECTORS)
