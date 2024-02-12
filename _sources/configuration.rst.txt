Configuration
=============

PPI.bio deployments are configured using ``.env`` files.

Here are the environment variables you can configure:

.. list-table:: Variables
   :widths: 15 85
   :header-rows: 1

   * - Variable
     - Description
   * - ``SECRET_KEY``
     - A secret key used for cryptographic purposes. Make this secure.
   * - ``DATABASE_URL``
     - A database URL that conforms to `this schema <https://github.com/jazzband/dj-database-url/blob/master/README.rst#url-schema>`_. This is the same 12Factor-style schema that Dokku and Heroku use.
   * - ``DEBUG``
     - Whether the website should be in Debug mode. In Debug mode, error stacktraces are reproduced. For security reasons, ``DEBUG`` should be ``False`` in Production.
   * - ``LOCALE_LANGUAGE_CODE``
     - The language code for the interface. Currently only supports and defaults to ``en-us``.
   * - ``LOCALE_TIME_ZONE``
     - The timezone the server is running on. Defaults to `UTC <https://en.wikipedia.org/wiki/Coordinated_Universal_Time>`_.
   * - ``Q_WORKERS``
     - Number of workers per instance of the job queue task runner, `Django Q <https://django-q.readthedocs.io/en/latest/>`_.
   * - ``Q_TIMEOUT``
     - Max time for a job to take before being retried (in seconds). Defaults to 60000.
   * - ``DJANGO_STATIC_HOST``
     - From what domain should static files be served. Only relevant when ``DEBUG=False``. Good for serving from CDNs.
   * - ``REDIS_URL``
     - A database URL that conforms to `this schema <https://github.com/jazzband/dj-database-url/blob/master/README.rst#url-schema>`_ pointing to a Redis server. This is the same 12Factor-style schema that Dokku and Heroku use.
   * - ``ALLOWED_HOSTS``
     - What domain should the site be served from. "localhost" is a good value for local development.
   * - ``MAILGUN_PRIVATE_KEY``
     - The `Mailgun <https://www.mailgun.com/>`_ private key, used for sending newsletter emails.
   * - ``MAILGUN_WEBHOOK_SIGNING_KEY``
     - The `Mailgun <https://www.mailgun.com/>`_ webhook key, used for sending newsletter emails.
   * - ``DEFAULT_FROM_EMAIL``
     - Where emails are sent from
   * - ``MAILGUN_SENDER_DOMAIN``
     - The domain from which Mailgun will send emails.