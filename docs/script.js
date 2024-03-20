const btnSendPhoto = document.querySelector('#btnSendPhoto')
const inpImg = document.querySelector('#inpImg')
const divContentFiles = document.querySelector('#contentFiles')

const addElementOcr = (key, value) => `
    <div>
        <h3>${key}</h3>
        <p>${value}</p>
    </div>
`

btnSendPhoto.addEventListener('click', e => {
    e.preventDefault()
    const formData = new FormData()
    Array.from(inpImg.files).forEach(file => {
        formData.append('image', file)
    })
    fetch(window.origin + '/ocr/', {
        method: 'POST',
        body: formData
    })
        .then(resp => resp.json())
        .then(ocr => {
            divContentFiles.innerHTML = Object
                .entries(ocr)
                .map(([key, value]) => addElementOcr(key, value))
                .join('')
        })
})