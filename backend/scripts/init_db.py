# backend/scripts/init_db.py
from contextshield.db.session import get_default_engine, init_db


def main():
    engine = get_default_engine()
    init_db(engine)
    print("Database tables created.")


if __name__ == "__main__":
    main()
