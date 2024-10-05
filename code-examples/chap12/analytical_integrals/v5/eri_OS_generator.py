import jinja2
from functools import lru_cache
from basis import iter_cart_xyz

c_tpl = jinja2.Template('''
void {{ func_name }}(double *eri, double ai, double aj, double ak, double al,
        double *Ra, double *Rb, double *Rc, double *Rd,
        void (*gamma_inc_fn)(int, double, double []))
{
    extern double exp(double);
    extern double sqrt(double);
    double aij = ai + aj;
    double akl = ak + al;
    double theta = aij * akl / (aij + akl);
    double Xab = Ra[0] - Rb[0];
    double Yab = Ra[1] - Rb[1];
    double Zab = Ra[2] - Rb[2];
    double Xcd = Rc[0] - Rd[0];
    double Ycd = Rc[1] - Rd[1];
    double Zcd = Rc[2] - Rd[2];
    double theta_rij = ai * aj / aij * (Xab*Xab+Yab*Yab+Zab*Zab);
    double theta_rkl = ak * al / akl * (Xcd*Xcd+Ycd*Ycd+Zcd*Zcd);
    double fac = 34.986836655249725; // 2*pi**2.5
    double Kabcd = fac / (aij*akl*sqrt(aij+akl)) * exp(-theta_rij - theta_rkl);
    if (Kabcd < 1e-16) {
        for (int i = 0; i < {{ nf }}; i++) eri[i] = 0;
        return;
    }
    double Xp = (ai * Ra[0] + aj * Rb[0]) / aij;
    double Yp = (ai * Ra[1] + aj * Rb[1]) / aij;
    double Zp = (ai * Ra[2] + aj * Rb[2]) / aij;
    double Xq = (ak * Rc[0] + al * Rd[0]) / akl;
    double Yq = (ak * Rc[1] + al * Rd[1]) / akl;
    double Zq = (ak * Rc[2] + al * Rd[2]) / akl;
    double Xpq = Xp - Xq;
    double Ypq = Yp - Yq;
    double Zpq = Zp - Zq;
    double Xpa = Xp - Ra[0];
    double Ypa = Yp - Ra[1];
    double Zpa = Zp - Ra[2];
    double Xqc = Xq - Rc[0];
    double Yqc = Yq - Rc[1];
    double Zqc = Zq - Rc[2];
    double theta_r2 = theta * (Xpq*Xpq+Ypq*Ypq+Zpq*Zpq);
    double theta_aij = theta / aij;
    double theta_akl = theta / akl;
    double aij_akl = aij / akl;
    double aa = aij + akl;
    double Xtheta_aij = Xpq * theta_aij;
    double Ytheta_aij = Ypq * theta_aij;
    double Ztheta_aij = Zpq * theta_aij;
    double Xtheta_akl = Xpq * theta_akl;
    double Ytheta_akl = Ypq * theta_akl;
    double Ztheta_akl = Zpq * theta_akl;
    double _gamma_inc[32];
    gamma_inc_fn({{ n_max+1 }}, theta_r2, _gamma_inc);
{{ code | indent(width=4, first=True) }}
}''')

def primitive_ERI(li, lj, lk, ll):
    ixyz = iter_cart_xyz(li)
    jxyz = iter_cart_xyz(lj)
    kxyz = iter_cart_xyz(lk)
    lxyz = iter_cart_xyz(ll)
    nfi = len(ixyz)
    nfj = len(jxyz)
    nfk = len(kxyz)
    nfl = len(lxyz)

    code = []
    n_max = 0

    # Based on the _primitive_ERI function in v1
    @lru_cache(10000)
    def vrr(n, ix, iy, iz):
        nonlocal n_max
        n_max = max(n_max, n)

        val = f'vrr_{n}_{ix}{iy}{iz}'
        if iz > 0:
            code.append(f'double {val} = Zpa*({vrr(n, ix, iy, iz-1)}) - Ztheta_aij*({vrr(n+1, ix, iy, iz-1)});')
            if iz > 1:
                code.append(f'{val} += {(iz-1)*.5}/aij * (({vrr(n, ix, iy, iz-2)}) - theta_aij*({vrr(n+1, ix, iy, iz-2)}));')
            return val

        if iy > 0:
            code.append(f'double {val} = Ypa*({vrr(n, ix, iy-1, iz)}) - Ytheta_aij*({vrr(n+1, ix, iy-1, iz)});')
            if iy > 1:
                code.append(f'{val} += {(iy-1)*.5}/aij * (({vrr(n, ix, iy-2, iz)}) - theta_aij*({vrr(n+1, ix, iy-2, iz)}));')
            return val

        if ix > 0:
            code.append(f'double {val} = Xpa*(({vrr(n, ix-1, iy, iz)})) - Xtheta_aij*(({vrr(n+1, ix-1, iy, iz)}));')
            if ix > 1:
                code.append(f'{val} += {(ix-1)*.5}/aij * (({vrr(n, ix-2, iy, iz)}) - theta_aij*({vrr(n+1, ix-2, iy, iz)}));')
            return val

        code.append(f'double {val} = Kabcd * _gamma_inc[{n}];')
        return val

    @lru_cache(10000)
    def trr(n, ix, iy, iz, kx, ky, kz):
        val = f'e_trans{n}_{ix}{iy}{iz}_{kx}{ky}{kz}'
        if kz > 0:
            code.append(f'double {val} = Zqc * ({trr(n, ix, iy, iz, kx, ky, kz-1)}) + Ztheta_akl * ({trr(n+1, ix, iy, iz, kx, ky, kz-1)});')
            if kz > 1:
                code.append(f'{val} += {(kz-1)*.5}/akl * ({trr(n, ix, iy, iz, kx, ky, kz-2)} - theta_akl*({trr(n+1, ix, iy, iz, kx, ky, kz-2)}));')
            if iz > 0:
                code.append(f'{val} += {iz*.5}/aa * ({trr(n+1, ix, iy, iz-1, kx, ky, kz-1)});')
            return val

        if ky > 0:
            code.append(f'double {val} = Yqc * ({trr(n, ix, iy, iz, kx, ky-1, kz)}) + Ytheta_akl * ({trr(n+1, ix, iy, iz, kx, ky-1, kz)});')
            if ky > 1:
                code.append(f'{val} += {(ky-1)*.5}/akl * ({trr(n, ix, iy, iz, kx, ky-2, kz)} - theta_akl*({trr(n+1, ix, iy, iz, kx, ky-2, kz)}));')
            if iy > 0:
                code.append(f'{val} += {iy*.5}/aa * ({trr(n+1, ix, iy-1, iz, kx, ky-1, kz)});')
            return val

        if kx > 0:
            code.append(f'double {val} = Xqc * ({trr(n, ix, iy, iz, kx-1, ky, kz)}) + Xtheta_akl * ({trr(n+1, ix, iy, iz, kx-1, ky, kz)});')
            if kx > 1:
                code.append(f'{val} += {(kx-1)*.5}/akl * ({trr(n, ix, iy, iz, kx-2, ky, kz)} - theta_akl*({trr(n+1, ix, iy, iz, kx-2, ky, kz)}));')
            if ix > 0:
                code.append(f'{val} += {ix*.5}/aa * ({trr(n+1, ix-1, iy, iz, kx-1, ky, kz)});')
            return val

        return vrr(n, ix, iy, iz)

    @lru_cache(10000)
    def hrr(ix, iy, iz, jx, jy, jz, kx, ky, kz, lx, ly, lz):
        val = f'hrr_{ix}{iy}{iz}_{jx}{jy}{jz}_{kx}{ky}{kz}_{lx}{ly}{lz}'
        if lz > 0:
            code.append(f'double {val} = ({hrr(ix, iy, iz, jx, jy, jz, kx, ky, kz+1, lx, ly, lz-1)}) + Zcd * ({hrr(ix, iy, iz, jx, jy, jz, kx, ky, kz, lx, ly, lz-1)});')
            return val
        if ly > 0:
            code.append(f'double {val} = ({hrr(ix, iy, iz, jx, jy, jz, kx, ky+1, kz, lx, ly-1, lz)}) + Ycd * ({hrr(ix, iy, iz, jx, jy, jz, kx, ky, kz, lx, ly-1, lz)});')
            return val
        if lx > 0:
            code.append(f'double {val} = ({hrr(ix, iy, iz, jx, jy, jz, kx+1, ky, kz, lx-1, ly, lz)}) + Xcd * ({hrr(ix, iy, iz, jx, jy, jz, kx, ky, kz, lx-1, ly, lz)});')
            return val
        if jz > 0:
            code.append(f'double {val} = ({hrr(ix, iy, iz+1, jx, jy, jz-1, kx, ky, kz, lx, ly, lz)}) + Zab * ({hrr(ix, iy, iz, jx, jy, jz-1, kx, ky, kz, lx, ly, lz)});')
            return val
        if jy > 0:
            code.append(f'double {val} = ({hrr(ix, iy+1, iz, jx, jy-1, jz, kx, ky, kz, lx, ly, lz)}) + Yab * ({hrr(ix, iy, iz, jx, jy-1, jz, kx, ky, kz, lx, ly, lz)});')
            return val
        if jx > 0:
            code.append(f'double {val} = ({hrr(ix+1, iy, iz, jx-1, jy, jz, kx, ky, kz, lx, ly, lz)}) + Xab * ({hrr(ix, iy, iz, jx-1, jy, jz, kx, ky, kz, lx, ly, lz)});')
            return val
        return trr(0, ix, iy, iz, kx, ky, kz)

    eris = []
    for i, (ix, iy, iz) in enumerate(ixyz):
        for j, (jx, jy, jz) in enumerate(jxyz):
            for k, (kx, ky, kz) in enumerate(kxyz):
                for l, (lx, ly, lz) in enumerate(lxyz):
                    # Collect all "eri[..] = ...;" in a list and dump them together later
                    eris.append(hrr(ix, iy, iz, jx, jy, jz, kx, ky, kz, lx, ly, lz))
    for n, val in enumerate(eris):
        code.append(f'eri[{n}] = {val};')

    return c_tpl.render(func_name=f'run_eri_{li}{lj}{lk}{ll}',
                        nf=nfi*nfj*nfk*nfl, n_max=n_max, code='\n'.join(code))


if __name__ == '__main__':
    driver_tpl = jinja2.Template('''
int run_eri_unrolled(double *eri, int li, int lj, int lk, int ll,
        double ai, double aj, double ak, double al,
        double *Ra, double *Rb, double *Rc, double *Rd,
        void (*gamma_inc_fn)(int, double, double []))
{
    int ijkl = li*{{ (lmax+1)**3 }} + lj*{{ (lmax+1)**2 }} + lk*{{ (lmax+1) }} + ll;
    switch (ijkl) {
{%- for li in range(lmax) %}
{%- for lj in range(lmax) %}
{%- for lk in range(lmax) %}
{%- for ll in range(lmax) %}
{%- if li + lj + lk + ll <= max_roots %}
{%- set ijkl = li*(lmax+1)**3 + lj*(lmax+1)**2 + lk*(lmax+1) + ll %}
    case {{ ijkl }}: run_eri_{{ li }}{{ lj }}{{ lk }}{{ ll }}(eri, ai, aj, ak, al, Ra, Rb, Rc, Rd, gamma_inc_fn); break;
{%- endif %}
{%- endfor %}
{%- endfor %}
{%- endfor %}
{%- endfor %}
    default: return 1; }
    return 0;
}''')

    lmax = 5
    max_roots = 5
    for li in range(lmax):
        for lj in range(lmax):
            for lk in range(lmax):
                for ll in range(lmax):
                    if li + lj + lk + ll > max_roots:
                        continue
                    print(primitive_ERI(li, lj, lk, ll))

    print(driver_tpl.render(lmax=lmax, max_roots=max_roots))
