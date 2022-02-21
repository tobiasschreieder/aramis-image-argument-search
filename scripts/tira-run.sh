#!/bin/bash
docker run -v /media:/media -v /home/touche22-aramis/working:/working aramis-imarg:latest -w /working -f "$@"