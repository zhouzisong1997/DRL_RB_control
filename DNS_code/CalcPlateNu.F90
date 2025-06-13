!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                         ! 
!    FILE: CalcPlateNu.F90                                !
!    CONTAINS: subroutine CalcPlateNu                     !
!                                                         ! 
!    PURPOSE: Calculate the Nusselt number at the top     !
!     and bottom plates and output to a file.             !
!                                                         !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine CalcPlateNu
      use param
      use local_arrays, only: temp
      use mpih
      use decomp_2d, only: xstart,xend
      implicit none
      integer :: j,i
      real ::  nuslow, nusupp
      real :: del,deln

      character(len=256) :: file_path
  

      nuslow = 0.d0
      nusupp = 0.d0
      del  = 1.0/(xc(2)-xc(1))
      deln = 1.0/(xc(nx)-xc(nxm))

!$OMP  PARALLEL DO &
!$OMP   DEFAULT(none) &
!$OMP   SHARED(xstart,xend,temp,del,deln) &
!$OMP   SHARED(nxm,nx) &
!$OMP   PRIVATE(i,j) &
!$OMP   REDUCTION(+:nuslow) &
!$OMP   REDUCTION(+:nusupp)
      do i=xstart(3),xend(3)
         do j=xstart(2),xend(2)
           nuslow = nuslow + (temp(1,j,i)-temp(2,j,i))*del
           nusupp = nusupp + (temp(nxm,j,i)-temp(nx,j,i))*deln
        enddo
      end do
!$OMP END PARALLEL DO

      nuslow = nuslow / (nzm*nym)
      nusupp = nusupp / (nzm*nym)

      call MpiSumRealScalar(nuslow)
      call MpiSumRealScalar(nusupp)

      if(ismaster) then
           write(file_path, '(a)') trim(local_tmpdir) // '/nu_plate.out'
           open(98,file=file_path,status='unknown', &
            access='sequential',position='append')
           write(98,546) time, nuslow, nusupp
 546   format(3(1x,e14.6))
           close(98)
      endif




      return         
      end                                                               




