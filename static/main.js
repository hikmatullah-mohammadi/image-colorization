img_input = document.getElementById('img-inp')
err_message = document.getElementById('err-not-grayscale')
img_input.addEventListener('change', () => {
  try {err_message.innerHTML = ''}
  catch {return}
})