# copernicusmarine in a slim Python image, for mirroring CMEMS GlobColour daily
# data to a Synology NAS (DS1821+, x86-64) on a schedule.
#
# Build on the NAS (SSH in, then):
#   cd /volume1/<share>/Copernicus-GlobColour/deploy/synology
#   docker build -t cmems-mirror .
#
# Run one sync (DSM Task Scheduler triggers this weekly):
#   docker run --rm --name cmems_mirror \
#     --env-file /volume1/<share>/Copernicus-GlobColour/cmems.env \
#     -v /volume1/<share>/Copernicus-GlobColour:/data \
#     cmems-mirror
FROM python:3.12-slim

RUN pip install --no-cache-dir copernicusmarine

COPY mirror_sync.sh /usr/local/bin/mirror_sync.sh
RUN chmod +x /usr/local/bin/mirror_sync.sh

ENTRYPOINT ["/usr/local/bin/mirror_sync.sh"]
