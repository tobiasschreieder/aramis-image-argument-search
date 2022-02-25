#!/bin/bash
docker run -v /media:/media -v /tmp:/tmp -v /home/touche22-aramis/working:/working aramis-imarg:latest -w /working -f "$@"