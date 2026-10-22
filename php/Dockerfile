# PHP runtime able to boot the MyHordes Symfony kernel, used only to export game data.
FROM php:8.4-cli

COPY --from=mlocati/php-extension-installer:2 /usr/bin/install-php-extensions /usr/local/bin/
COPY --from=composer:2 /usr/bin/composer /usr/local/bin/composer

RUN apt-get update \
 && apt-get install -y --no-install-recommends git unzip \
 && rm -rf /var/lib/apt/lists/* \
 && install-php-extensions intl redis zip \
 && echo 'memory_limit=-1' > /usr/local/etc/php/conf.d/memory.ini \
 && git config --global --add safe.directory '*'

ENV COMPOSER_ALLOW_SUPERUSER=1
WORKDIR /app
