const = 1


def dict_to_message(d):
    result = []
    for key, value in d.items():
        values = [str(value["value"]), str(value["x"]), str(value["y"]), str(value["w"]), str(value["h"]),
                  str(value["coef"])]
        result.append(", ".join(values))
    return "\n" + "\n".join(result)


def check_area(prediction):
    try:
        for ID in range(len(prediction) - 1):
            if prediction.at[ID, 'h'] * prediction.at[ID, 'w'] > const:
                prediction.drop(index=ID)
        return prediction
    except TypeError:
        return TypeError
