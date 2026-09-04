from api.main import app  # re-export so `fastapi dev` / `fastapi run` find it here


def main():
    print("Hello from revenue-forecasting!")


if __name__ == "__main__":
    main()
