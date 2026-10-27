"""Small helpers for reading the sim state."""

_geom_ids = {}


def _ball_contact_ids(model):
    """(ball, blade, floor) geom ids, looked up by name once per model."""
    key = id(model)
    if key not in _geom_ids:
        _geom_ids[key] = (model.geom("ball").id, model.geom("blade").id, model.geom("floor").id)
    return _geom_ids[key]


def detect_ball_contact(model, data):
    """Return (on_paddle, on_floor): whether the ball is touching the paddle blade / the floor
    at this step. Both come from MuJoCo's contact list, which mj_step refreshes every step."""
    ball, blade, floor = _ball_contact_ids(model)
    on_paddle = on_floor = False
    for a, b in zip(data.contact.geom1[: data.ncon], data.contact.geom2[: data.ncon]):
        if a != ball and b != ball:
            continue
        other = b if a == ball else a
        if other == blade:
            on_paddle = True
        elif other == floor:
            on_floor = True
    return on_paddle, on_floor
