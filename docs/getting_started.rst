Getting Started
===============

Let's get our feet wet by running PPI.org on our local computers. This runs just fine on my
`Thinkpad T495 <http://web.archive.org/web/20230524214704/https://www.lenovo.com/ca/en/p/laptops/thinkpad/thinkpadt/t495/22tp2ttt495>`_
laptop, so you don't need a supercomputer to run it.

You will, however, need to have the following things installed.

- `PostgresSQL <https://www.postgresql.org/download/>`_
- `Redis <https://redis.io/download/#redis-downloads>`_
- `Python 3.8 or greater <https://www.python.org/downloads/>`_
- `pip <https://pip.pypa.io/en/stable/installation/>`_
- (Optional) `virtualenv <https://virtualenv.pypa.io/en/latest/installation.html>`_

Step One - Clone the Repo
-------------------------

.. code-block:: bash

    git clone https://github.com/jszym/ppi.bio ppi_bio

Step Two - Set-up your environment
----------------------------------

Optionally begin by creating a virtual environment for RAPPPID Online

.. code-block:: bash

    python -m virtualenv venv
    source venv/bin/activate

Install all the required packages using pip.

.. code-block:: bash

    cd ppi_bio
    python -m pip install -r requirements.txt

Step Three - Configure your ``.env`` file
-----------------------------------------

We need to configure the local deployment using a ``.env`` located in the ``rapppid_org`` directory.

There is a handy example ``.env`` file distributed with the PPI.bio code base. We can begin by
copying as so:

.. code-block:: bash

    cp ppi_bio/.env.example ppi_bio/.env

Consult the `configuration <configuration.html>`_ documentation for what to change in this file.

Step Four - Migrate the Database
--------------------------------
Next, we need to migrate our database, which will create all the tables we need. To do so, run

.. code-block:: bash

    python manage.py migrate

Step Five - Create your admin account
-------------------------------------
We'll be using the admin interface in a minute, so we'll need to create an admin account to log into. That's easy to do
with the following one-liner:

.. code-block:: bash

    python manage.py createsuperuser

Then, just follow the on-screen instructions.

Step Six - Run the job runner
-----------------------------

RAPPPID Online requires that, in a separate process, the `Django Q <https://django-q.readthedocs.io/en/latest/>`_
job runner is running to process all the long-running ML tasks.

Open up another terminal window/tab and run the following:

.. code-block:: bash

    python manage.py qcluster

Step Seven - Seed the database
------------------------------

.. warning::
    This can take a long time, depending on the species.

We will need to download and load the UniProt cDNA protein sequences for the organisms you wish to make proteome
predictions. Type in the following command:

.. code-block:: bash

    python manage.py proteome_seed 9606

9606 is the NCBI taxon code for humans, and so this command seeds the database with human protein sequences. To do so
for other sequences, you will first need to create an ``Organism`` object that corresponds to your desired species. You
can do this in the admin panel.


Step Eight - Precompute the INTREPPPID/RAPPPID embeddings
---------------------------------------------------------
.. warning::
    This is even longer than the previous step. It can take even longer than a day on under-powered laptops or servers.
    What I've had to do on `PPI.bio <https://ppi.bio>`_ is compute the embeddings on a faster desktop computer, dump
    the database to an SQL file, and then restore the database on the PPI.bio server.

.. warning::
    Do not run this before the last step is complete. Otherwise, it'll quit early and you'll have a bunch of proteins
    without their corresponding INTREPPPID/RAPPPID vectors.

In order to make proteome-wide predictions, PPI.bio will need to pre-compute the vectors for the protein sequences in
the database. To do this, run the following command:

.. code-block:: bash

    python manage.py proteome_embed 9606 nest-much

This will pre-compute the embeddings for all the human sequences using the "nest-much" weights. The "nest-much" weights
come installed out-of-the-box, and are trained on Human PPI data from the v11.5 of the
`STRING database <https://string-db.org/>`_.

You can install other INTREPPPID/RAPPPID weights by adding a new Weights object through the Django Admin panel.

Step Six - Run the test server
------------------------------

You can run the test server locally with the following command:

.. code-block:: bash

    python manage.py runserver

This will output a URL that you can visit to go to your PPI.bio server.