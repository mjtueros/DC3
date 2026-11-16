"""Optional bridge to the external RF-chain code (``apply_rfchain``).

The E-field emulator needs nothing from it. Only the E-field -> voltage step does
(``to_voltage``, ``predict_voltage``, ...). So a missing ``apply_rfchain`` must not
break ``import pulse_emulator``: the four functions below are replaced by stubs that
raise a clear ImportError when they are *called*.

``apply_rfchain`` lives in https://github.com/arsenefer/RFchain_computation, which is
not pip-installable; clone it and put it on PYTHONPATH.
"""
try:
    from apply_rfchain import (
        efield_2_voltage,
        make_full_response_matrix,
        percieved_theta_phi,
        voltage_to_adc,
    )
    HAS_RFCHAIN = True
except ImportError:
    HAS_RFCHAIN = False

    def _missing(name):
        def stub(*args, **kwargs):
            raise ImportError(
                f"{name}() needs the external 'apply_rfchain' module to convert E-fields "
                "to voltages. Clone https://github.com/arsenefer/RFchain_computation and add "
                "it to PYTHONPATH. E-field predictions work without it."
            )
        stub.__name__ = name
        return stub

    percieved_theta_phi = _missing("percieved_theta_phi")
    efield_2_voltage = _missing("efield_2_voltage")
    make_full_response_matrix = _missing("make_full_response_matrix")
    voltage_to_adc = _missing("voltage_to_adc")


def require_rfchain():
    """Raise early (before heavy work) if apply_rfchain is unavailable."""
    if not HAS_RFCHAIN:
        percieved_theta_phi(None, None)
