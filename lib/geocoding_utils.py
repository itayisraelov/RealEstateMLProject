import json
import urllib.request
import urllib.parse
from lib.constants import *


def geocode(address, key):
    """
    Return a json dictionary with data on the given address
    :param address: The address you wish to acquire data on
    :type address: str
    :param key: Your geocode Key
    :type key: str
    :exception: Exception if something went wrong
    :return: Json dictionary with data on given address
    :rtype: Result example:
    {
    'address_components': [{
            'long_name': '625',
            'short_name': '625',
            'types': ['street_number']
        }, {
            'long_name': 'Rogers Avenue',
            'short_name': 'Rogers Ave',
            'types': ['route']
        }, {
            'long_name': 'Prospect Lefferts Gardens',
            'short_name': 'Prospect Lefferts Gardens',
            'types': ['neighborhood', 'political']
        }, {
            'long_name': 'Brooklyn',
            'short_name': 'Brooklyn',
            'types': ['political', 'sublocality', 'sublocality_level_1']
        }, {
            'long_name': 'Kings County',
            'short_name': 'Kings County',
            'types': ['administrative_area_level_2', 'political']
        }, {
            'long_name': 'New York',
            'short_name': 'NY',
            'types': ['administrative_area_level_1', 'political']
        }, {
            'long_name': 'United States',
            'short_name': 'US',
            'types': ['country', 'political']
        }, {
            'long_name': '11225',
            'short_name': '11225',
            'types': ['postal_code']
        }, {
            'long_name': '3809',
            'short_name': '3809',
            'types': ['postal_code_suffix']
        }
    ],
    'formatted_address': '625 Rogers Ave, Brooklyn, NY 11225, USA',
    'geometry': {
        'bounds': {
            'northeast': {
                'lat': 40.656369,
                'lng': -73.9527311
            },
            'southwest': {
                'lat': 40.6562886,
                'lng': -73.95293989999999
            }
        },
        'location': {
            'lat': 40.6563206,
            'lng': -73.9528586
        },
        'location_type': 'ROOFTOP',
        'viewport': {
            'northeast': {
                'lat': 40.65767778029149,
                'lng': -73.9514865197085
            },
            'southwest': {
                'lat': 40.65497981970849,
                'lng': -73.9541844802915
            }
        }
    },
    'place_id': 'ChIJ4VLvT2lbwokRD_dB5215Si8',
    'types': ['premise']
}

    """
    # Join the parts of the URL together into one string.
    params = urllib.parse.urlencode({"address": address, "key": key})
    url = f"{GEOCODE_BASE_URL}?{params}"

    result = json.load(urllib.request.urlopen(url))

    if result["status"] in ["OK", "ZERO_RESULTS"]:
        return result["results"]

    raise Exception(result["error_message"])


def get_lat_lng(address, key):
    """
    Get the latitude and longitude for a given address using google geocoding API
    :param address: The address to query on
    :type address: str
    :param key: google geocoding API Key
    :type key: str
    :return: The latitude and longitude of the given address
    :rtype: (float, float)
    :exception: Exception if something went wrong
    """
    geocode_res = geocode(address, key)
    lat = geocode_res["geometry"]["location"]['lat']
    lng = geocode_res["geometry"]["location"]['lng']
    return lat, lng


def get_near_by_search(lat, lng, radius, search_type, key):
    """
    Query geocode for nearby places in a certain radius, around the given
    latitude and longitude.
    :param search_type: The type of places to query around the given location,
    for available search types please see:

    https://developers.google.com/places/web-service/supported_types

    :type search_type: str
    :param lat:
    :type lat: float
    :param lng:
    :type lng: float
    :param radius: Radius limit for the query
    :type radius: float
    :param key: Google Places API  Key
    :type key: str
    :return: A list of the nearby places that match the query, each represented
    like the json bellow
    :rtype: A list of places each represented by the below json dictionary
    {
        'business_status': 'OPERATIONAL',
        'geometry': {
            'location': {
                'lat': 40.6435287,
                'lng': -73.7797708
            },
            'viewport': {
                'northeast': {
                    'lat': 40.64484598029149,
                    'lng': -73.7784464697085
                },
                'southwest': {
                    'lat': 40.64214801970849,
                    'lng': -73.7811444302915
                }
            }
        },
        'icon': 'https://maps.gstatic.com/mapfiles/place_api/icons/cafe-71.png',
        'name': "Peet's Coffee",
        'opening_hours': {
            'open_now': True
        },
        'photos': [{
                'height': 3024,
                'html_attributions': ['<a href="https://maps.google.com/maps/contrib/105095935235207447551">Veronika Kiryanova</a>'],
                'photo_reference': 'CmRaAAAAI2lgOtjL4mrqdr7_Ox64Rfhk3WQh5jqOvvIW2F8wPumX-1ZWuq7DQiuo-DJdldXQLw3Ce6BZzhzL7F1QVdV9K9u82FFNcsVrSH63uuHuKQ8TDW-2hXB1OsNbVAOlV18rEhAvJlZBIAuq29kgR-Oz5D_cGhQeEvufI2jFaVEl608iavlJx3HVwg',
                'width': 4032
            }
        ],
        'place_id': 'ChIJ_eQYpFdmwokRgdc96drmvVQ',
        'plus_code': {
            'compound_code': 'J6VC+C3 New York, NY, USA',
            'global_code': '87G8J6VC+C3'
        },
        'price_level': 1,
        'rating': 3.6,
        'reference': 'ChIJ_eQYpFdmwokRgdc96drmvVQ',
        'scope': 'GOOGLE',
        'types': ['cafe', 'food', 'point_of_interest', 'store', 'establishment'],
        'user_ratings_total': 144,
        'vicinity': 'Jamaica'
    }
    """
    location = f"{lat},{lng}"
    radius = radius
    # Join the parts of the URL together into one string.
    params = urllib.parse.urlencode(
        {
            "query": "NYCRealEstateProjectQuery",
            "opennow": "false",
            "location": location,
            "radius": radius,
            "type": search_type,
            "key": key,
        }
    )
    url = f"{URL_NEARYBY_SEARCH}?{params}"
    result = json.load(urllib.request.urlopen(url))

    if result["status"] in ["OK", "ZERO_RESULTS"]:
        return result["results"]

    raise Exception(result["error_message"])
