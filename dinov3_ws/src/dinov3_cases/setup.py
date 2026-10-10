from setuptools import setup

setup(
    name="dinov3_cases",
    version="0.2.0",
    packages=["dinov3_cases"],
    package_data={"dinov3_cases": ["templates/*.html", "static/*.js", "static/*.css"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/dinov3_cases"]),
        ("share/dinov3_cases", ["package.xml"]),
    ],
    entry_points={"console_scripts": ["case_recognition = dinov3_cases.node:main"]},
)
