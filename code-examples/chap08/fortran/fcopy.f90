subroutine dcopy(len, inArray, outArray)
  integer*4, intent(in) :: len
  real*8, intent(in) :: inArray(*)
  real*8, intent(out) :: outArray(*)
  integer :: i
  do i = 1, len
    outArray(i) = inArray(i)
  end do
end subroutine

subroutine strcopy(inStr, outStr)
  character(*), intent(in) :: inStr
  character(*), intent(out) :: outStr
  print*, inStr
  outStr = inStr
end subroutine
