from setuptools import find_packages, setup

setup(
    name="dinov3_cases",
    version="0.2.0",
    packages=find_packages(include=["dinov3_cases", "dinov3_cases.*"]),
    package_data={"dinov3_cases.api": ["templates/*.html", "static/*.js", "static/*.css"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/dinov3_cases"]),
        ("share/dinov3_cases", ["package.xml"]),
    ],
    entry_points={"console_scripts": ["case_recognition = dinov3_cases.ros.node:main"]},
)
