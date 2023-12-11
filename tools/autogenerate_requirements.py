import toml


def main():
    with open("pyproject.toml", "r") as f:
        pyproject = toml.load(f)

    with open("requirements.txt", "w") as f:
        for dep in pyproject["project"]["dependencies"]:
            if dep.startswith("python"):
                continue
            f.write(dep + "\n")

        for name, deps in pyproject["project"]["optional-dependencies"].items():
            f.write(f"\n# {name}\n")
            for dep in deps:
                if dep.startswith("datadreamer"):
                    continue
                f.write(dep + "\n")


if __name__ == "__main__":
    main()
