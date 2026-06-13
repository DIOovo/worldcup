import requests


class OpenMeteoClient:
    """Open-Meteo 天气数据客户端。"""

    def __init__(self):
        self.geocoding_url = (
            "https://geocoding-api.open-meteo.com/v1/search"
        )
        self.forecast_url = (
            "https://api.open-meteo.com/v1/forecast"
        )

    def search_city(
        self,
        city_name: str,
        country_code: str | None = None
    ):
        """
        根据城市名称查询经纬度。

        city_name:
            城市名称，例如 New York。

        country_code:
            国家代码，例如 US。
        """

        params = {
            "name": city_name,
            "count": 10,
            "language": "en",
            "format": "json"
        }

        if country_code:
            params["countryCode"] = country_code

        response = requests.get(
            self.geocoding_url,
            params=params,
            timeout=20
        )

        response.raise_for_status()

        return response.json()

    def find_city(
        self,
        city_name: str,
        country_code: str
    ):
        """
        查询并返回最匹配的城市。

        使用国家代码避免同名城市匹配错误。
        """

        data = self.search_city(
            city_name=city_name,
            country_code=country_code
        )

        results = data.get("results", [])

        if not results:
            raise ValueError(
                f"没有找到城市：{city_name}, {country_code}"
            )

        return results[0]

    def get_hourly_weather(
        self,
        latitude: float,
        longitude: float,
        forecast_days: int = 7
    ):
        """
        获取指定经纬度未来几天的逐小时天气。
        """

        params = {
            "latitude": latitude,
            "longitude": longitude,

            "hourly": ",".join([
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation_probability",
                "precipitation",
                "weather_code",
                "wind_speed_10m",
                "wind_gusts_10m"
            ]),

            "timezone": "auto",
            "forecast_days": forecast_days
        }

        response = requests.get(
            self.forecast_url,
            params=params,
            timeout=20
        )

        response.raise_for_status()

        return response.json()


def print_hourly_weather(weather_data, limit: int = 24):
    """打印前 limit 个小时的天气数据。"""

    hourly = weather_data.get("hourly", {})

    times = hourly.get("time", [])
    temperatures = hourly.get("temperature_2m", [])
    humidity = hourly.get("relative_humidity_2m", [])
    rain_probability = hourly.get(
        "precipitation_probability",
        []
    )
    precipitation = hourly.get("precipitation", [])
    weather_codes = hourly.get("weather_code", [])
    wind_speeds = hourly.get("wind_speed_10m", [])
    wind_gusts = hourly.get("wind_gusts_10m", [])

    count = min(limit, len(times))

    for index in range(count):
        print("-" * 70)
        print("时间：", times[index])
        print("温度：", temperatures[index], "°C")
        print("湿度：", humidity[index], "%")
        print("降水概率：", rain_probability[index], "%")
        print("降水量：", precipitation[index], "mm")
        print("天气代码：", weather_codes[index])
        print("风速：", wind_speeds[index], "km/h")
        print("阵风：", wind_gusts[index], "km/h")


if __name__ == "__main__":
    client = OpenMeteoClient()

    # 使用城市名和国家代码查找美国纽约
    city = client.find_city(
        city_name="New York",
        country_code="US"
    )

    print("匹配到的城市：")
    print("城市：", city.get("name"))
    print("国家：", city.get("country"))
    print("国家代码：", city.get("country_code"))
    print("纬度：", city.get("latitude"))
    print("经度：", city.get("longitude"))
    print("时区：", city.get("timezone"))
    print("=" * 70)

    weather_data = client.get_hourly_weather(
        latitude=city["latitude"],
        longitude=city["longitude"],
        forecast_days=7
    )

    print("天气接口返回时区：", weather_data.get("timezone"))
    print("海拔：", weather_data.get("elevation"))
    print("=" * 70)

    print_hourly_weather(
        weather_data=weather_data,
        limit=24
    )