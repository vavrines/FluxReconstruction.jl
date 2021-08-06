"""
    neighbor_fpidx(IDs, ps, fpg)

global id
local rank

"""
function neighbor_fpidx(IDs, ps, fpg)
    # id-th cell, fd-th face, jd-th point
    id, fd, jd = IDs

    # ending point ids of a face
    if fd == 1
        pids = [ps.cellid[id, 1], ps.cellid[id, 2]]
    elseif fd == 2
        pids = [ps.cellid[id, 2], ps.cellid[id, 3]]
    elseif fd == 3
        pids = [ps.cellid[id, 3], ps.cellid[id, 1]]
    end

    # global face index
    faceids = ps.cellFaces[id, :]

    function get_faceid()
        for i in eachindex(faceids)
            if sort(pids) == sort(ps.facePoints[faceids[i], :])
                return faceids[i]
            end
        end

        @warn "no face id found"
    end
    faceid = get_faceid()

    # neighbor cell id
    neighbor_cid = setdiff(ps.faceCells[faceid, :], id)[1]

    # in case of boundary cell
    if neighbor_cid <= 0
        return neighbor_cid, -1, -1
    end

    # face rank in neighbor cell
    if ps.cellid[neighbor_cid, 1] ∉ ps.facePoints[faceid, :]
        neighbor_frk = 2
    elseif ps.cellid[neighbor_cid, 2] ∉ ps.facePoints[faceid, :]
        neighbor_frk = 3
    elseif ps.cellid[neighbor_cid, 3] ∉ ps.facePoints[faceid, :]
        neighbor_frk = 1
    end

    # point rank in neighbor cell
    neighbor_nrk1 =
        findall(x -> x == fpg[id, fd, jd, 1], fpg[neighbor_cid, neighbor_frk, :, 1])
    neighbor_nrk2 =
        findall(x -> x == fpg[id, fd, jd, 2], fpg[neighbor_cid, neighbor_frk, :, 2])
    neighbor_nrk = intersect(neighbor_nrk1, neighbor_nrk2)[1]

    return neighbor_cid, neighbor_frk, neighbor_nrk
end
