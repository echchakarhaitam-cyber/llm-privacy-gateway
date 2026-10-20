import pytest

from privgate.detectors.validators import iban_valid, ipv4_valid, luhn_valid


@pytest.mark.parametrize(
    "digits, expected",
    [
        ("4111111111111111", True),  # classic Visa test number
        ("4111111111111112", False),  # last digit changed
        ("79927398713", True),  # textbook Luhn example
        ("79927398710", False),
        ("0", False),  # too short
        ("", False),
        ("4111-1111-1111-1111", False),  # validator expects digits only
        ("４１１１１１１１１１１１１１１１", False),  # full-width digits are not ASCII
    ],
)
def test_luhn(digits, expected):
    assert luhn_valid(digits) is expected


@pytest.mark.parametrize(
    "iban, expected",
    [
        ("GB82 WEST 1234 5698 7654 32", True),  # ISO 13616 example
        ("GB82WEST12345698765432", True),
        ("gb82 west 1234 5698 7654 32", True),  # case-insensitive
        ("GB82 WEST 1234 5698 7654 33", False),  # checksum broken
        ("GB82 WEST", False),  # too short
        ("1234 5678 9012 3456", False),  # no country code
        ("", False),
    ],
)
def test_iban(iban, expected):
    assert iban_valid(iban) is expected


@pytest.mark.parametrize(
    "ip, expected",
    [
        ("192.168.0.1", True),
        ("0.0.0.0", True),
        ("255.255.255.255", True),
        ("999.1.1.1", False),
        ("256.1.1.1", False),
        ("1.2.3", False),
        ("1.2.3.4.5", False),
        ("01.2.3.4", False),  # leading zero rejected
        ("1..3.4", False),
        ("a.b.c.d", False),
    ],
)
def test_ipv4(ip, expected):
    assert ipv4_valid(ip) is expected
