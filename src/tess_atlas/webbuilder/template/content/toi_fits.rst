TOI FITs
=========

We analysed {{number['total']}} TOIs:

- {{number['done']}} TOIs successfully analysed
- {{number['fail']}} TOIs had errors

Each TOI's analysis has its own page.
Here we have a summary of all the TOI fits and links to their pages.

Successful Fits
---------------

..
  each item in listtable will be a TOI number and the phase plots

Normal Systems
^^^^^^^^^^^^^^^

{{number['norm_done']}}/{{number['norm']}} normal TOIs completed execution.

.. raw:: html

   <details>
   <summary><a>big code</a></summary>

    .. list-table::
        :widths: 5 15
        :header-rows: 1
        :stub-columns: 1
        {% for toi_link, images in successful_tois['norm'].items() %}
        * - {{ toi_link }}
          - {% for image in images %} {{ image }}
            {% endfor %}
        {% endfor %}

.. raw:: html

   </details>

Multi-planet Systems
^^^^^^^^^^^^^^^^^^^^

{{number['multi_done']}}/{{number['multi']}} multi-planet TOIs completed execution.

.. raw:: html

   <details>
   <summary><a>big code</a></summary>

    .. list-table::
        :widths: 5 15
        :header-rows: 1
        :stub-columns: 1
        {% for toi_link, images in successful_tois['multi'].items() %}
        * - {{ toi_link }}
          - {% for image in images %} {{ image }}
            {% endfor %}
        {% endfor %}

.. raw:: html

   </details>

Single-transit Systems
^^^^^^^^^^^^^^^^^^^^^^

{{number['single_done']}}/{{number['single']}} single-transit TOIs completed execution.

.. raw:: html

   <details>
   <summary><a>big code</a></summary>

    .. list-table::
        :widths: 5 15
        :header-rows: 1
        :stub-columns: 1
        {% for toi_link, images in successful_tois['single'].items() %}
        * - {{ toi_link }}
          - {% for image in images %} {{ image }}
            {% endfor %}
        {% endfor %}

.. raw:: html

   </details>

Erroneous fits
---------------
The failed TOIs:

- {{number['norm_fail']}}/{{number['norm']}} normal TOIs
- {{number['multi_fail']}}/{{number['multi']}} multi-planet TOIs
- {{number['single_fail']}}/{{number['single']}} single-transit TOIs

..
  list with links to erroneous fits


Normal Systems
^^^^^^^^^^^^^^^

.. raw:: html

   <details>
   <summary><a>big code</a></summary>

    {% for toi_link in failed_tois['norm'] %}
    - {{ toi_link }}
    {% endfor %}

 .. raw:: html

    </details>

Multi-planet Systems
^^^^^^^^^^^^^^^^^^^^

.. raw:: html

   <details>
   <summary><a>big code</a></summary>

    {% for toi_link in failed_tois['multi'] %}
    - {{ toi_link }}
    {% endfor %}

 .. raw:: html

    </details>

Single-transit Systems
^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

   <details>
   <summary><a>big code</a></summary>

    {% for toi_link in failed_tois['single'] %}
    - {{ toi_link }}
    {% endfor %}

 .. raw:: html

    </details>
