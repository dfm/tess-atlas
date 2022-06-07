TOI FITs
=========

We analysed {{total_number_tois}} TOIs:

- {{number_successful_tois}} TOIs successfully analysed
- {{number_failed_tois}} TOIs had errors

Each TOI's analysis has its own page.
Here we have a summary of all the TOI fits and links to their pages.

Successful Fits
---------------

..
  each item in listtable will be a TOI number and the phase plots

.. list-table::
    :widths: 5 15
    :header-rows: 1
    :stub-columns: 1
    {% for toi_link, images in successful_tois.items() %}
    * - {{ toi_link }}
      - {% for image in images %} {{ image }}
        {% endfor %}
    {% endfor %}




Erroneous fits
---------------

..
  list with links to erroneous fits

{% for toi_link in failed_tois %}
- {{ toi_link }}
{% endfor %}
