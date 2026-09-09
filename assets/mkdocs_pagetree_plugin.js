// SPDX-FileCopyrightText: 2023 Thomas Breitner
//
// SPDX-License-Identifier: MIT

/**
* Collapse/expand all pagetree <details> sections
*/
function toggleDetails (pagetree, state) {
  const pagetreeDetails = pagetree.querySelectorAll('.pagetree details')

  pagetreeDetails.forEach(pagetreeDetail => {
    if (state === 'expand') {
      pagetreeDetail.setAttribute('open', 'open')
    } else {
      pagetreeDetail.removeAttribute('open')
    }
  })
}

/**
* Insert Collapse/expand button
*/
function insertCollapseExpandButton (pagetreeContainerElement, pagetreeElement, pagetreeFunctionsElement) {

  // Return early if there are no toggleable elements
  const hasToggleElems = pagetreeElement.querySelector("details");
  if (!hasToggleElems) return;

  const toggleBtn = '<button class="pagetree-toggle md-button btn btn-primary btn-sm my-2" type="button">Expand/Collapse</button>'

  pagetreeFunctionsElement.insertAdjacentHTML('afterbegin', toggleBtn)

  pagetreeContainerElement.querySelectorAll('.pagetree-toggle').forEach(button => {
    let btnClickCount = 0
    button.addEventListener('click', (e) => {
      btnClickCount++
      if (btnClickCount % 2 === 1) {
        toggleDetails(pagetreeElement, 'expand')
      } else {
        toggleDetails(pagetreeElement, 'collapse')
      }
    })
  })
}

function getPagestatusSelect (pageStatusesArray) {
  const pagestatusSelectElem = `
  <select name="pagestatus-select" id="pagestatus-select" class="pagestatus-select md-button btn btn-primary btn-sm my-2">
    <option value="">Filter by page status</option>
    ${pageStatusesArray.map(status => `
      <option value="${status}">Page status: ${status}</option>
    `).join('\n')}
  </select>
  `
  return pagestatusSelectElem
}

function insertPageStatusFilter (pagetreeContainerElement, pagetreeElement, pagetreeFunctionsElement) {
  const pagetreeStatusMarker = '.pagetree-pagestatus'
  const pagesWithPagestatus = pagetreeContainerElement.querySelectorAll(pagetreeStatusMarker)

  // Not interesseted in duplicate values, so using a set...
  let pageStatuses = new Set()
  for (const pageStatus of pagesWithPagestatus) {
    pageStatuses.add(pageStatus.dataset.pageStatus)
  };

  if (pageStatuses.size) {
    // Only render/expose the filter function if any page status exists
    // ...and convert the set back to an array
    pageStatuses = Array.from(pageStatuses)

    const pagestatusSelect = getPagestatusSelect(pageStatuses)
    pagetreeFunctionsElement.insertAdjacentHTML('beforeend', pagestatusSelect)

    const pageStatusSelectElement = document.getElementById('pagestatus-select')
    pageStatusSelectElement.addEventListener('click', (e) => {
    })
    pageStatusSelectElement.addEventListener('change', (event) => {
      filterTree(pagetreeElement, event.target.value)
    })
  }
}

function filterTree (pagetreeElement, pagestatus) {
  // console.info(`Filtering for page status: ${pagestatus}`);

  const pagetreeItems = pagetreeElement.querySelectorAll('.pagetree-navlink')
  for (const pagetreeItem of pagetreeItems) {
    const closestLi = pagetreeItem.closest('li')

    if (!pagestatus) {
      // no pagestatus given, reset pagetree
      closestLi.style.display = 'list-item'
    } else if (pagetreeItem.dataset.pageStatus !== pagestatus) {
      closestLi.style.display = 'none'
    } else {
      closestLi.style.display = 'list-item'
    }
  }
  toggleDetails(pagetreeElement, 'expand')
}

document.addEventListener('DOMContentLoaded', function () {
  const pagetreeContainerElement = document.querySelector('.pagetree-container')

  // Return early if target element not found
  if (!pagetreeContainerElement) return;

  const pagetreeFunctionsElement = pagetreeContainerElement.querySelector('.pagetree-functions')
  const pagetreeElement = pagetreeContainerElement.querySelector('.pagetree')

  insertCollapseExpandButton(pagetreeContainerElement, pagetreeElement, pagetreeFunctionsElement)
  insertPageStatusFilter(pagetreeContainerElement, pagetreeElement, pagetreeFunctionsElement)
})
