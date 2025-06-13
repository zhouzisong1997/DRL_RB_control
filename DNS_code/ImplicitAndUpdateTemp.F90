!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                         ! 
!    FILE: ImplicitAndUpdateTemp.F90                      !
!    CONTAINS: subroutine ImplicitAndUpdateTemp           !
!                                                         ! 
!    PURPOSE: Compute the linear terms associated to      !
!     the temperature and call the implicit solver.       !
!     After this routine, the temperature has been        !
!     updated to the new timestep                         !
!                                                         !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine ImplicitAndUpdateTemp
      use param
      use local_arrays, only: temp,hro,rutemp,rhs
      use decomp_2d, only: xstart,xend
      implicit none
      integer :: jc,kc,ic
      integer :: km,kp
      real    :: alpec,dxxt
      real    :: app,acc,amm

      alpec=al/pec

!$OMP  PARALLEL DO &
!$OMP   DEFAULT(none) &
!$OMP   SHARED(xstart,xend,nxm,temp) &
!$OMP   SHARED(kmv,kpv,am3ck,ac3ck,ap3ck) &
!$OMP   SHARED(ga,ro,alpec,dt) &
!$OMP   SHARED(rhs,rutemp,hro) &
!$OMP   PRIVATE(ic,jc,kc,km,kp) &
!$OMP   PRIVATE(amm,acc,app) &
!$OMP   PRIVATE(dxxt)
      do ic=xstart(3),xend(3)
      do jc=xstart(2),xend(2)
      do kc=2,nxm

!   Calculate second derivative of temperature in the x-direction.
!   This is the only term calculated implicitly for temperature.

               dxxt= temp(kc+1,jc,ic)*ap3ck(kc) &
                    +temp(kc  ,jc,ic)*ac3ck(kc) &
                    +temp(kc-1,jc,ic)*am3ck(kc)


!    Calculate right hand side of Eq. 5 (VO96)

            rhs(kc,jc,ic)=(ga*hro(kc,jc,ic)+ro*rutemp(kc,jc,ic) &
                    +alpec*dxxt)*dt

!    Store the non-linear terms for the calculation of 
!    the next timestep

            rutemp(kc,jc,ic)=hro(kc,jc,ic)

        enddo
       enddo
      enddo
!$OMP END PARALLEL DO


!  Solve equation and update temperature

      call SolveImpEqnUpdate_Temp

!  Set boundary conditions on the temperature field at top
!  and bottom plates. This seems necessary.

       !temp(1,xstart(2):xend(2),xstart(3):xend(3)) &
       !   = tempbp(xstart(2):xend(2),xstart(3):xend(3))

       temp(nx,xstart(2):xend(2),xstart(3):xend(3)) &
          = temptp(xstart(2):xend(2),xstart(3):xend(3))


      return
      end subroutine ImplicitAndUpdateTemp



 
 
 
 
 
 
 
 
      subroutine Temp_BC
            use param
            use local_arrays, only: temp
            use mpih
            use decomp_2d, only: xstart,xend
            implicit none
            integer :: j,i
            integer :: k_ref
            real :: vol00



      !      if(ismaster) then
      !       open(97,file="alpha0000.out",status='unknown', &
      !        access='sequential',position='append')
      !       write(97,549) alpha_ac, beta_ac, alpha_ac_final
      ! 549   format(3(1x,e14.6))
      !       close(97)
      !      endif



            do i=xstart(3),xend(3)
               do j=xstart(2),xend(2)
                     temp(1,j,i)=T_ac_low(j,i)+1.0d0
                     !temp(nx,j,i)=T_ac_upp(j,i)
                     !temp(nx,j,i)=0.0d0
               enddo
            enddo

            return         
      end subroutine Temp_BC  
      
      subroutine ReadACParameters0
            use param
            use local_arrays, only: temp
            use mpih
            use decomp_2d, only: xstart,xend
            implicit none
            integer :: k,j,i
            integer :: i00,j00,k00
            integer :: k_ref
            integer :: len_total
            integer :: io_status
            real ::  temp_avg_low1, temp_rms_low1, temp_avg_low2, temp_rms_low2
            !real ::  temp_avg_upp1, temp_rms_upp1, temp_avg_upp2, temp_rms_upp2
            real ::  alpha_0, alpha_15
            real :: vol00
            real, allocatable, dimension(:) :: action_org, action_total
            character*30 filename0,dsetname0
            character*70 :: filnam
            logical :: file_noexists,file_noexists1,file_exists

            !allocate(T_ac_low(nym,nzm),T_ac_upp(nym,nzm))
            allocate(T_ac_low(nym,nzm))


            if(ismaster) then
                  open(unit=33, file='tmpdir.txt', status='old', action='read', iostat=io_status)
                  read(unit=33, fmt='(A)', iostat=io_status) local_tmpdir
            else
            endif
            call MPI_BCAST(local_tmpdir, 100, MPI_CHARACTER, 0, MPI_COMM_WORLD, ierr)


            ! if(ismaster) then
            !       open(unit=10, file='b1.txt', status='replace', action='write')
            !       write(10, *) "LOCAL_TMPDIR in Fortran: ", trim(local_tmpdir)
            !       close(10)
            ! else
            ! endif




      
            return         
      end subroutine ReadACParameters0        



      subroutine ReadACParameters
            use param
            use local_arrays, only: temp
            use mpih
            use decomp_2d, only: xstart,xend
            implicit none
            integer :: k,j,i
            integer :: i00,j00,k00
            integer :: k_ref
            integer :: len_total
            real ::  temp_avg_low1, temp_rms_low1, temp_avg_low2, temp_rms_low2, temp_avg_low3, temp_rms_low3, temp_max_low1
            !real ::  temp_avg_upp1, temp_rms_upp1, temp_avg_upp2, temp_rms_upp2
            real ::  alpha_0, alpha_15
            real :: vol00
            real, allocatable, dimension(:) :: action_org, action_total
            character*200 filename0,dsetname0
            character*70 :: filnam
            character(len=256) :: file_path,file_path1,file_path2
            logical :: file_noexists,file_noexists1,file_exists

            !allocate(T_ac_low(nym,nzm),T_ac_upp(nym,nzm))
            ! allocate(T_ac_low(nym,nzm))



            


            len_total=nzm*nym


            !!!!!
            vol00 = 1.d0/(real(nzm)*real(nym))
            k_ref=14




            !!!!
            write(file_path, '(a)') trim(local_tmpdir) // '/continua.dat'
            file_noexists = .true.
            call MPI_BARRIER(MPI_COMM_WORLD,ierr)


            

             if(ismaster) then
                  file_noexists = .true.
                  do while (file_noexists)
                        INQUIRE(FILE=file_path, EXIST=file_exists)
                        file_noexists = .not. file_exists
                  end do

                   open (29, file=file_path, form='unformatted')
                   close (29, status='delete')
             else
             endif


            


             call MPI_Barrier(MPI_COMM_WORLD, ierr)




             write(file_path1, '(a)') trim(local_tmpdir) // '/T_action.h5'


            if(ismaster) then
                  allocate(action_org(len_total),action_total(len_total))

                  filename0 = trim(file_path1)
                  dsetname0 = trim('action')
                  call HdfSerialReadReal1D00(dsetname0,filename0,action_org,len_total)



                  action_total=action_org





                  

   
                  temp_avg_low2 = 0.0d0
                  do i=1,len_total
                        temp_avg_low2=temp_avg_low2+action_total(i)
                  enddo
                  temp_avg_low2=temp_avg_low2/real(len_total)

                  
                  do i=1,len_total
                        action_total(i)=action_total(i)-temp_avg_low2
                  enddo

                  temp_max_low1=0.5d0

                  do i=1,len_total
                        temp_max_low1=max(abs(action_total(i)),temp_max_low1)
                  enddo

                  do i=1,len_total
                        action_total(i)=action_total(i)/temp_max_low1
                  enddo



                  





                  do i=1,len_total
                        j00=mod(i,nym)
                        if(j00==0) then
                              j00=nym
                        else
                        endif
                        k00=(i-j00)/nym+1
                        T_ac_low(j00,k00)=action_total(i)
                  enddo







            ! 550   format(' ',6e17.9)
            !       open(unit=26,file='alpha_state.dat',status='unknown',access='sequential',position='append')
            !       write(26,550) temp_rms_low1,temp_rms_low2,temp_rms_low3,temp_avg_low1,temp_avg_low2,temp_avg_low3

            !       close(26)

                  ! alpha_0=max(temp_rms_upp2,0.001d0)
                  ! alpha_15=min(temp_rms_upp1,0.1d0)

                  ! do j=1,nym
                  !       do k=1,nzm
                  !             T_ac_upp(j,k)=T_ac_upp(j,k)-temp_avg_upp2
                  !             T_ac_upp(j,k)=T_ac_upp(j,k)*alpha_15/alpha_0

                  !             i00=i00+1
                  !             action_total(i00)=action_total(i00)-temp_avg_upp2
                  !             action_total(i00)=action_total(i00)*alpha_15/alpha_0
                  !       enddo
                  ! enddo


                  ! call HdfSerialWriteReal1D00(dsetname0,filename0,action_total,len_total)




                  !!!!!!!!!
                  !open(97,file="alpha0000.out",status='unknown', access='sequential',position='append')
                  !write(97,549) temp_avg_low2, min(temp_rms_low1,0.1d0)/max(temp_rms_low2,0.001d0),temp_avg_upp2,max(temp_rms_upp2,0.001d0)/min(temp_rms_upp1,0.1d0)
                  !549   format(4(1x,e14.6))
                  !close(97)
                  !!!!!!!!!





            else
            endif


            call MPI_BCAST(T_ac_low,nzm*nym,MDP,0,MPI_COMM_WORLD,ierr)



      
            return         
      end subroutine ReadACParameters     























      subroutine WriteStates
            use param
            use local_arrays, only: temp
            use stat3_param
            use decomp_2d, only: xstart,xend
            implicit none
            integer :: i,j,m,k_ref,k,len_total,i00,j00,k00
            real ::  T_avg_low1
            real ::  vol00
            real, allocatable, dimension(:) :: temp_state1
            character(len=256) :: file_path,file_path1
            character*200 :: filename0,dsetname0

            ! character(len=512) :: command





            len_total=nzm*nym

            allocate(temp_state1(len_total))


            write(file_path, '(a)') trim(local_tmpdir) // '/T_state_low.h5'
            write(file_path1, '(a)') trim(local_tmpdir) // '/continua_py.dat'



            k_ref=14


            vol00 = 1.d0/(real(nzm)*real(nym))
            T_avg_low1 = 0.0d0
            do i=xstart(3),xend(3)
               do j=xstart(2),xend(2)
                  T_avg_low1 = T_avg_low1 + temp(1+k_ref,j,i)*vol00 
              enddo
            end do
            call MpiSumRealScalar(T_avg_low1)
            call MpiBcastReal(T_avg_low1)












            do i=1,len_total
                  temp_state1(i)=0.0d0
            enddo 

              do k00=xstart(3),xend(3)
               do j00=xstart(2),xend(2)
                  i00=(k00-1)*nym+j00
                 temp_state1(i00) = temp(1+k_ref,j00,k00)-T_avg_low1
                enddo
              enddo

              call MpiSumReal1D(temp_state1,len_total)

            if(ismaster) then
                  filename0 = trim(file_path)
                  dsetname0 = trim('state')
              call HdfCreateBlankFile00(filename0)
              call HdfSerialWriteReal1D00(dsetname0,filename0,temp_state1,len_total)

                        
            !   open(unit=10, file='b2.txt', status='replace', action='write')
            !   write(10, *) "LOCAL_TMPDIR in Fortran: ", filename0
            !   close(10)

            !   command = 'ls ' // trim(local_tmpdir) // '/ > aaaa_output.txt'
            !   call execute_command_line(command)

            !   command = 'cp ' // filename0 // ' /scratch-emmy/projects/nip00068/zhouzisong/DRL-TD3/DRL-DNS-test4/DNS_result/v_state_low.h5'
            !   call execute_command_line(command)






            else
            endif

            if(ismaster) then
                  open (30, file=file_path1, form='unformatted')
                  close(30)
            else
            endif

            




            !filnam='v_state_low'
            !call DumpStates(temp_state1,filnam)
            !filnam='v_state_upp'
            !call DumpStates(temp_state2,filnam)



            
            !!!!!! wall
            !do i=xstart(3),xend(3)
            !      do j=xstart(2),xend(2)
            !       temp_state1(j,i) = temp(1,j,i)
            !        temp_state2(j,i) = temp(nx,j,i)
            !      enddo
            !enddo
            !
            !filnam='v_state_low11'
            !call DumpStates(temp_state1,filnam)
            !filnam='v_state_upp11'
            !call DumpStates(temp_state2,filnam)


      
            return
      end subroutine WriteStates



      subroutine WriteStates_test(Istep)
            use param
            use local_arrays, only: temp
            use stat3_param
            use decomp_2d, only: xstart,xend
            implicit none
            integer :: Istep
            integer :: i,j,m,k_ref
            real,dimension(xstart(2):xend(2),xstart(3):xend(3)) :: temp_state1,temp_state2
            character*70 :: filnam1,filnam2
            character*8 :: numsave

            write(numsave,'(I8.8)') Istep
      
            



            k_ref=14

              do i=xstart(3),xend(3)
               do j=xstart(2),xend(2)
                 temp_state1(j,i) = temp(1+k_ref,j,i)
                 temp_state2(j,i) = temp(1,j,i)
                enddo
              enddo


            filnam1 = trim('T_state_'//numsave)
            call DumpStates(temp_state1,filnam1)
            filnam2 = trim('T_action_'//numsave)
            call DumpActions(temp_state2,filnam2)



            
            !!!!!! wall
            !do i=xstart(3),xend(3)
            !      do j=xstart(2),xend(2)
            !       temp_state1(j,i) = temp(1,j,i)
            !        temp_state2(j,i) = temp(nx,j,i)
            !      enddo
            !enddo
            !
            !filnam='T_state_low11'
            !call DumpStates(temp_state1,filnam)
            !filnam='T_state_upp11'
            !call DumpStates(temp_state2,filnam)


      
            return
      end subroutine WriteStates_test









      !==================================================================
      
      subroutine DumpStates(var,filnam)
            USE param
            use mpih
            USE hdf5
            use decomp_2d, only: xstart,xend
            IMPLICIT none
      
            real, intent(in) :: var(xstart(2):xend(2) &
           &                  ,xstart(3):xend(3))
      
      
            character*70,intent(in) :: filnam
            character*70 :: namfile,dsetname
      

      
            namfile=trim(trim(filnam)//'.h5')
            dsetname = trim('state')
      
            call HdfWriteReal2D(dsetname,namfile,var)
      
      
            return                                                          
      end subroutine DumpStates



      !==================================================================
      
      subroutine DumpActions(var,filnam)
            USE param
            use mpih
            USE hdf5
            use decomp_2d, only: xstart,xend
            IMPLICIT none
      
            real, intent(in) :: var(xstart(2):xend(2) &
           &                  ,xstart(3):xend(3))
      
      
            character*70,intent(in) :: filnam
            character*70 :: namfile,dsetname
      

      
            namfile=trim(trim(filnam)//'.h5')
            dsetname = trim('action')
      
            call HdfWriteReal2D(dsetname,namfile,var)
      
      
            return                                                          
      end subroutine DumpActions







!       subroutine Temp_BC1
!             use param
!             use local_arrays, only: temp
!             use mpih
!             use decomp_2d, only: xstart,xend
!             implicit none
!             integer :: j,i
!             integer :: k_ref
!             real ::  temp_avg_low, temp_avg_upp
!             real :: vol00

!             vol00 = 1.d0/(real(nzm)*real(nym))

!             !k_ref=nxm/2
!             k_ref=14
        
      
!             temp_avg_low = 0.d0
!             temp_avg_upp = 0.d0

!             do i=xstart(3),xend(3)
!                do j=xstart(2),xend(2)
!                   temp_avg_low = temp_avg_low + temp(1+k_ref,j,i)*vol00
!                   temp_avg_upp = temp_avg_upp + temp(nx-k_ref,j,i)*vol00
!               enddo
!             end do
      
      
!             call MpiSumRealScalar(temp_avg_low)
!             call MpiSumRealScalar(temp_avg_upp)

!             call MpiBcastReal(temp_avg_low)
!             call MpiBcastReal(temp_avg_upp)

!             do i=xstart(3),xend(3)
!                do j=xstart(2),xend(2)
!                      temp(1,j,i)=(temp(1+k_ref,j,i)-temp_avg_low)+1.0d0
!                      temp(nx,j,i)=(temp(nx-k_ref,j,i)-temp_avg_upp)
!                      !temp(nx,j,i)=0.0
!                enddo
!             end do

!    !         if(ismaster) then
!    !          open(97,file="vavg.out",status='unknown', &
!    !           access='sequential',position='append')
!    !          write(97,549) temp_avg_low, temp_avg_upp
!    !    549   format(2(1x,e14.6))
!    !          close(97)
!    !         endif
      

      
!             return         
!       end subroutine Temp_BC1     