import psycopg2


def get_encoding_map() -> dict:
    """
    Get map of encoding id -> name
    For each event token
    """
    c = psycopg2.connect("")
    cursor = c.cursor()

    cursor.execute(
        """
        --sql
        SELECT encoding, label, value FROM mimiciv_local.item_encoding;
        """
    )

    res = cursor.fetchall()
    map = {i[0]: f"{i[1]}: {i[2]}" for i in res}
    map[0] = "Null event"

    c.close()

    return map


if __name__ == "__main__":
    d = get_encoding_map()
    print(d)
